# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import dataclasses
import gc
import heapq
import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import torch

from semantic_parsing_with_constrained_lm.decoding.partial_parse import PartialParse
from semantic_parsing_with_constrained_lm.lm import HS, AutoregressiveModel
from semantic_parsing_with_constrained_lm.lm_openai_gpt3 import Instrumentation
from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer

PSNSub = TypeVar("PSNSub", bound="PackedSearchNode")


# https://github.com/python/mypy/issues/5374
@dataclass(frozen=True, eq=True)  # type: ignore
class PackedSearchNode(ABC):
    """Contains all state for SearchNode in compact form.

    ConstrainedDecodingProblem contains a cache for `expand`, with PackedSearchNode as the key.
    In order to start beam search from some arbitrary state, clients can construct PackedSearchNode cheaply
    (i.e. without actually running a neural network, or creating a PartialParse object).
    Then, if the PackedSearchNode is in the cache already, then we can look up its expansion cheaply.
    If it's not in the cache, `ConstrainedDecodingProblem.unpacker` can turn it into a SearchNode.
    """

    # Output tokens generated so far.
    tokens: Tuple[int, ...]

    @abstractmethod
    def append(self: PSNSub, token: int) -> PSNSub:
        pass

    def extend(self: PSNSub, tokens: Sequence[int]) -> PSNSub:
        result = self
        for token in tokens:
            result = result.append(token)
        return result


# TODO: Make a new class for finished search nodes?
# It can omit partial_parse and hidden_state.
@dataclass
class FullSearchNode(Generic[HS]):
    packed: PackedSearchNode

    partial_parse: PartialParse
    hidden_state: Optional[HS] = dataclasses.field(repr=False)

    is_finished: bool = False
    cost: float = 0
    unnormalized_cost: float = 0

    @property
    def tokens(self) -> Tuple[int, ...]:
        return self.packed.tokens

    # This function duplicates the above but its form is more convenient sometimes.
    def get_tokens(self) -> Tuple[int, ...]:
        return self.packed.tokens


SearchNode = Union[FullSearchNode[HS], PSNSub]

SearchNodeUnpacker = Callable[
    [PSNSub], Awaitable[Tuple[PartialParse, HS, Sequence[float]]]
]


class Problem(Generic[HS, PSNSub], ABC):
    """Defines the formal search problem used for decoding.

    Instances specify how to create successor search nodes given an existing
    one. A search algorithm, such as beam search, uses this interface to find
    suitable finished search nodes from an initial one.
    """

    @abstractmethod
    async def expand(
        self, maybe_packed_node: SearchNode[HS, PSNSub]
    ) -> List[FullSearchNode[HS]]:
        pass


@dataclass
class ConstrainedDecodingProblem(Problem[HS, PSNSub]):
    model: AutoregressiveModel[HS]
    # This function knows how to expand PackedSearchNodes.
    unpacker: SearchNodeUnpacker[PSNSub, HS]

    # A 1D Long tensor containing the IDs of tokens indicating the end.
    eos: torch.Tensor
    length_normalization: float
    # Only use the top_k most likely next tokens in `expand`.
    # This can be set to the beam size.
    top_k: Optional[int] = None

    # TODO: PackedSearchNode may not always be Hashable.
    cache: Optional[MutableMapping[PackedSearchNode, List[FullSearchNode[HS]]]] = None

    async def expand(
        self, maybe_packed_node: SearchNode[HS, PSNSub]
    ) -> List[FullSearchNode[HS]]:
        if self.cache is not None:
            if isinstance(maybe_packed_node, FullSearchNode):
                packed_node = maybe_packed_node.packed
            else:
                packed_node = maybe_packed_node
            existing = self.cache.get(packed_node)
            if existing is not None:
                logging.debug("\N{DIRECT HIT} %s", packed_node)
                return existing
            else:
                logging.debug("\N{HOURGLASS WITH FLOWING SAND} %s", packed_node)

        if isinstance(maybe_packed_node, FullSearchNode):
            assert not maybe_packed_node.is_finished
            assert maybe_packed_node.hidden_state
            logprobs, new_hidden_state = await self.model.extend(
                maybe_packed_node.tokens[-1:],
                maybe_packed_node.hidden_state,
            )

            next_logprobs = logprobs[0]  # Remove the sequence dimension
            unnormalized_cost = maybe_packed_node.unnormalized_cost
            packed_node = maybe_packed_node.packed
            partial_parse = maybe_packed_node.partial_parse
            # new_hidden_state already set
        else:
            (
                partial_parse,
                hidden_state,
                existing_logprobs,
            ) = await self.unpacker(  # type: ignore
                maybe_packed_node
            )

            next_logprobs = await self.model.next_logprobs(hidden_state)
            unnormalized_cost = -sum(existing_logprobs)
            packed_node = maybe_packed_node
            # partial_parse already set
            new_hidden_state = hidden_state

        del maybe_packed_node

        # Remove -inf entries
        mask = next_logprobs != -float("inf")
        ordered_tokens = torch.argsort(next_logprobs, descending=True)
        allowed_next, can_end = partial_parse.allowed_next(
            ordered_tokens[mask[ordered_tokens]], self.top_k
        )

        result: List[FullSearchNode[HS]] = []
        if can_end:
            eos_logprob = torch.logsumexp(next_logprobs[self.eos], dim=0)

            new_unnorm_cost = unnormalized_cost - eos_logprob.item()
            result.append(
                FullSearchNode(
                    packed_node,
                    partial_parse,
                    hidden_state=None,
                    is_finished=True,
                    cost=gnmt_length_normalization(
                        self.length_normalization,
                        new_unnorm_cost,
                        len(packed_node.tokens) + 1,
                    ),
                    unnormalized_cost=new_unnorm_cost,
                )
            )
        token_and_logprob_iter: Iterator[Tuple[int, torch.Tensor]]
        if allowed_next is None:
            indices = torch.arange(next_logprobs.shape[0])
            eligible_logprobs = next_logprobs
        else:
            indices = allowed_next
            eligible_logprobs = next_logprobs[allowed_next]

        if self.top_k is None:
            token_and_logprob_iter = (
                # .item() converts 0D tensor to a Python number
                (token_id_tensor.item(), logprob)  # type: ignore
                for token_id_tensor, logprob in zip(indices, eligible_logprobs)
            )
        else:
            topk_eligible_logprobs = torch.topk(
                eligible_logprobs,
                k=min(self.top_k, eligible_logprobs.shape[0]),
                sorted=False,
            )
            token_and_logprob_iter = (
                (token_id_tensor.item(), logprob)  # type: ignore
                for token_id_tensor, logprob in zip(
                    indices[topk_eligible_logprobs.indices],
                    topk_eligible_logprobs.values,
                )
            )

        for token, logprob in token_and_logprob_iter:
            if token in self.eos:
                continue
            new_unnorm_cost = unnormalized_cost - logprob.item()
            result.append(
                FullSearchNode(
                    packed_node.append(token),
                    partial_parse.append(token),
                    new_hidden_state,
                    cost=gnmt_length_normalization(
                        self.length_normalization,
                        new_unnorm_cost,
                        len(packed_node.tokens) + 1,
                    ),
                    unnormalized_cost=new_unnorm_cost,
                )
            )

        if self.cache is not None:
            self.cache[packed_node] = result
        return result


def gnmt_length_normalization(alpha: float, unnormalized: float, length: int) -> float:
    """
    Eq 14 from https://arxiv.org/abs/1609.08144, but missing the coverage term.
    TODO switch to something similar since this is probably overtuned for MT.
    :param unnormalized: log prob
    :param length: |Y|, or length of sequence.
    :return:
    """
    lp = (5 + length) ** alpha / (5 + 1) ** alpha
    return unnormalized / lp


@dataclass(frozen=True)
class BeamKey:
    tokens: Tuple[int, ...]
    is_finished: bool

    @classmethod
    def from_node(cls, node: FullSearchNode) -> "BeamKey":
        return cls(node.tokens, node.is_finished)


@dataclass
class HashableNodeWrapper(Generic[HS]):
    underlying: FullSearchNode[HS]
    _key: Tuple[Tuple[int, ...], bool] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self._key = (self.underlying.tokens, self.underlying.is_finished)

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, HashableNodeWrapper):
            return False
        return self._key == other._key


class BeamSearchEventListener:
    def step(
        self,
        expansions: Dict[
            Tuple[int, ...], Tuple[SearchNode[Any, Any], List[FullSearchNode[Any]]]
        ],
    ) -> None:
        pass


@dataclass
class LoggingEventListener(BeamSearchEventListener):
    tokenizer: ClampTokenizer
    beam_size: int
    last_step: int = 0

    def step(
        self,
        all_expansions: Dict[
            Tuple[int, ...], Tuple[SearchNode[Any, Any], List[FullSearchNode[Any]]]
        ],
    ) -> None:
        # TODO: Print which of the expansions are being kept in the beam/finished lists.
        already_printed: Set[HashableNodeWrapper] = set()
        header = f"===== DEPTH {self.last_step} ====="
        print(header)
        Instrumentation.print_last_requests()
        for _, (node, expansions) in all_expansions.items():
            if isinstance(node, FullSearchNode):
                node_cost_str = f" [{node.cost:.3f}]"
            else:
                node_cost_str = ""

            duplicates = 0
            # `node` expands to the nodes in `expansions`.
            print(
                f"Completions for {self.tokenizer.decode(list(node.tokens))!r}{node_cost_str}:"
            )
            for expansion in heapq.nsmallest(
                self.beam_size * 2, expansions, key=lambda n: n.cost
            ):
                deduplicating_node = HashableNodeWrapper(expansion)
                if deduplicating_node in already_printed:
                    duplicates += 1
                    continue
                already_printed.add(deduplicating_node)

                if expansion.is_finished:
                    complete = self.tokenizer.decode(list(expansion.tokens))
                    print(f"- Finished: {complete!r} -> [{expansion.cost:.3f}]")
                else:
                    last_token = self.tokenizer.decode(
                        list(expansion.tokens[len(node.tokens) :])
                    )
                    if isinstance(node, FullSearchNode):
                        print(
                            f"- {last_token!r} [{expansion.unnormalized_cost - node.unnormalized_cost:.3f}] -> [{expansion.cost:.3f}]"
                        )
                    else:
                        # TODO: Get the cost of the node so that we can report the difference
                        print(f"- {last_token!r} -> [{expansion.cost:.3f}]")

            if duplicates:
                print(f"- [{duplicates} duplicates of already printed in depth]")
            if len(expansions) > self.beam_size * 2:
                print(f"... and {len(expansions) - self.beam_size * 2} more")
        print("=" * len(header))
        self.last_step += 1


MAX_STEPS = 1000


async def beam_search(
    problem: Problem[HS, PSNSub],
    initial: SearchNode[HS, PSNSub],
    beam_size: int,
    max_steps: Optional[int] = None,
    event_listener: BeamSearchEventListener = BeamSearchEventListener(),
    keep_finished_nodes: bool = False,
) -> List[FullSearchNode[HS]]:
    max_steps = MAX_STEPS if max_steps is None else max_steps

    finished: Set[HashableNodeWrapper[HS]] = set()
    finished_extra: Set[HashableNodeWrapper[HS]] = set()

    beam: List[SearchNode[HS, PSNSub]] = [initial]

    for step_index in range(max_steps):
        if not beam:
            break

        async def expand(
            node: SearchNode[HS, PSNSub]
        ) -> Tuple[SearchNode[HS, PSNSub], List[FullSearchNode]]:
            expansions = await problem.expand(node)
            return node, expansions

        candidates: Set[HashableNodeWrapper[HS]] = set()
        step_info: Dict[
            Tuple[int, ...], Tuple[SearchNode[HS, PSNSub], List[FullSearchNode[HS]]]
        ] = {}
        for node, per_node_expansion in await asyncio.gather(
            *(expand(node) for node in beam)
        ):
            candidates_for_node: List[FullSearchNode] = []
            packed_node = node.packed if isinstance(node, FullSearchNode) else node
            step_info[packed_node.tokens] = (node, candidates_for_node)
            for new_node in per_node_expansion:
                candidates_for_node.append(new_node)
                candidates.add(HashableNodeWrapper(new_node))
        event_listener.step(step_info)

        # We allow `candidates` and `finished` to compete with each other,
        # as the score will no longer decrease monotonically when we have a length penalty.
        sorted_candidates_plus_finished = sorted(
            itertools.chain(candidates, finished), key=lambda n: n.underlying.cost
        )
        beam = []
        finished.clear()
        for n in sorted_candidates_plus_finished[:beam_size]:
            if n.underlying.is_finished:
                finished.add(n)
            else:
                beam.append(n.underlying)

        # If there's a less-competitive candidate which is finished, then keep it for later
        if keep_finished_nodes:
            for n in sorted_candidates_plus_finished[beam_size:]:
                if n.underlying.is_finished:
                    finished_extra.add(n)

        # Due to cycles or some other reason, hidden states are not freed on
        # time unless we manually collect.
        if step_index % 50 == 0 and step_index > 0:
            print("Garbage collecting ...")
            gc.collect()
            torch.cuda.empty_cache()

    print("Garbage collecting ...")
    gc.collect()
    torch.cuda.empty_cache()

    return sorted(
        (x.underlying for x in itertools.chain(finished, finished_extra)),
        key=lambda n: n.cost,
    )[: beam_size * 2]
