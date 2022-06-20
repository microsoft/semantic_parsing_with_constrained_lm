# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, cast

import torch

from semantic_parsing_with_constrained_lm.decoding.partial_parse import NullPartialParse, PartialParse
from semantic_parsing_with_constrained_lm.lm import TokensWithLogprobs
from semantic_parsing_with_constrained_lm.lm_openai_gpt3 import IncrementalOpenAIGPT3, OpenAIGPT3State
from semantic_parsing_with_constrained_lm.search import (
    ConstrainedDecodingProblem,
    FullSearchNode,
    Problem,
    PSNSub,
    SearchNode,
    SearchNodeUnpacker,
    gnmt_length_normalization,
)


@dataclass
class SpeculativeConstrainedDecodingProblem(Problem[OpenAIGPT3State, PSNSub]):
    # TODO: Support models other than IncrementalOpenAIGPT3, including use of hidden states
    model: IncrementalOpenAIGPT3
    unpacker: SearchNodeUnpacker[PSNSub, OpenAIGPT3State]

    length_normalization: float

    # Below are parameters for the GPT-3 completions API.
    # TODO: Put these into a closure together with `model`.
    eos_ids: torch.Tensor
    max_length: int
    num_completions: int
    temperature: float = 1
    top_p: float = 1

    eos: str = dataclasses.field(init=False)
    constrained_decoding_problem: ConstrainedDecodingProblem = dataclasses.field(
        init=False
    )

    def __post_init__(self):
        # Reverse-engineer the stop string from the eos_ids.
        # It is the longest common prefix of the tokens.
        # TODO: Get it as a string instead?
        eos_tokens = [
            self.model.tokenizer.id_to_utf8_token_map[i] for i in self.eos_ids.tolist()
        ]
        for i in range(min(len(s) for s in eos_tokens), 0, -1):
            if all(eos_tokens[0][:i] == s[:i] for s in eos_tokens[1:]):
                eos_bytes = eos_tokens[0][:i]
                break
        else:
            raise Exception("Could not infer EOS string from eos_tokens: {eos_tokens}")
        # May raise UnicodeDecodeError. Trim eos_bytes to whole Unicode code points to avoid this?
        self.eos = eos_bytes.decode("utf-8")

        self.constrained_decoding_problem = ConstrainedDecodingProblem(
            self.model,
            self.unpacker,
            self.eos_ids,
            self.length_normalization,
            self.num_completions,
        )

    async def expand(
        self, maybe_packed_node: SearchNode[OpenAIGPT3State, PSNSub]
    ) -> List[FullSearchNode[OpenAIGPT3State]]:
        if len(maybe_packed_node.tokens) >= self.max_length:
            return []

        max_completion_tokens = self.max_length - len(maybe_packed_node.tokens)
        if isinstance(maybe_packed_node, FullSearchNode):
            samples = await self.model.completions(
                maybe_packed_node.tokens[-1:],
                max_completion_tokens,
                self.temperature,
                self.top_p,
                self.num_completions,
                self.eos,
                hidden_state=maybe_packed_node.hidden_state,
            )
            unnormalized_cost = maybe_packed_node.unnormalized_cost
            base_partial_parse = maybe_packed_node.partial_parse
            packed_node = maybe_packed_node.packed
        else:
            (
                base_partial_parse,
                hidden_state,
                existing_logprobs,
            ) = await self.unpacker(  # type: ignore
                maybe_packed_node
            )

            samples = await self.model.completions(
                (),
                max_completion_tokens,
                self.temperature,
                self.top_p,
                self.num_completions,
                self.eos,
                hidden_state=hidden_state,
            )
            unnormalized_cost = -sum(existing_logprobs)
            # base_partial_parse already set
            packed_node = maybe_packed_node

        finished: List[TokensWithLogprobs] = []
        incomplete: List[Tuple[TokensWithLogprobs, OpenAIGPT3State, PartialParse]] = []

        for twl, new_hidden_state in samples:
            assert twl.token_ids.dim() == 1
            assert twl.token_ids.shape == twl.logprobs.shape

            partial_parse = base_partial_parse
            for offset in range(len(twl.token_ids)):
                # Check that the token at the offset is allowed
                allowed_next, can_end = partial_parse.allowed_next(
                    twl.token_ids[offset : offset + 1], top_k=1
                )
                token_id = cast(int, twl.token_ids[offset].item())

                if allowed_next is None or token_id in allowed_next:
                    partial_parse = partial_parse.append(token_id)
                    # Let's hope that the EOS token can't occur in the middle of
                    # a grammatical output.
                    continue

                if can_end:
                    # Decide whether we should consider the hypothesis finished.
                    # - we have an EOS token, the model indicating that we're done: yes
                    # - we don't have an EOS token
                    #   - PartialParse says we must end here: yes
                    #   - PartialParse says we can continue: no
                    if token_id in self.eos_ids:
                        # - we have an EOS token, the model indicating that we're done
                        add_to_finished = True
                    else:
                        # - we don't have an EOS token
                        any_allowed_next, _ = partial_parse.allowed_next(
                            torch.arange(self.model.tokenizer.vocab_size), top_k=1
                        )
                        # - True: PartialParse says we must end here
                        # - False: PartialParse says we can continue
                        add_to_finished = (
                            any_allowed_next is not None and len(any_allowed_next) == 0
                        )

                    if add_to_finished:
                        # TODO: Use the logsumexp of eos_ids logprobs?
                        finished.append(
                            TokensWithLogprobs(
                                twl.token_ids[: offset + 1], twl.logprobs[: offset + 1]
                            )
                        )
                        # Theoretically we might not want to break here if the EOS
                        # token is allowed to occur in the middle of a grammatical
                        # output.
                        break

                if offset > 0:
                    incomplete.append(
                        (
                            TokensWithLogprobs(
                                twl.token_ids[:offset], twl.logprobs[:offset]
                            ),
                            # TODO: Implement and instead use a truncate method on the hidden state class
                            OpenAIGPT3State(
                                new_hidden_state.tokens[
                                    : len(new_hidden_state.tokens)
                                    - (len(twl.token_ids) - offset)
                                ]
                            ),
                            partial_parse,
                        )
                    )
                break
            else:
                # We come here if there was no `break`.
                # In other words, all tokens in the completion were valid according to the grammar,
                # but we did not get an EOS token at a place where we are allowed to end.
                # Check here whether we are allowed to end.
                next_logprobs, _ = await self.model.extend(
                    twl.token_ids[-1:].tolist(),
                    new_hidden_state,
                    drop_next_hidden_state=True,
                )
                # [0]: remove the sequence dimension
                next_logprobs = next_logprobs[0]

                mask = next_logprobs != -float("inf")
                ordered_tokens = torch.argsort(next_logprobs, descending=True)

                # partial_parse should contain all tokens from earlier
                allowed_next, can_end = partial_parse.allowed_next(
                    ordered_tokens[mask[ordered_tokens]], top_k=self.num_completions
                )
                if can_end:
                    # eos_logprob is a 1D tensor with 1 element
                    eos_logprob = torch.logsumexp(next_logprobs[self.eos_ids], dim=0)

                    finished.append(
                        TokensWithLogprobs(
                            torch.cat((twl.token_ids, self.eos_ids[:1])),
                            torch.cat((twl.logprobs, eos_logprob.reshape(1))),
                        )
                    )
                else:
                    # GPT-3 decided to stop sampling before drawing an EOS
                    # token. Check whether it's because we reached the length limit.
                    if len(twl.token_ids) == max_completion_tokens:
                        pass
                    else:
                        raise Exception(
                            "Model unexpectedly stopped before the EOS token"
                        )
            del partial_parse

        result: List[FullSearchNode[OpenAIGPT3State]] = []
        for twl in finished:
            new_unnorm_cost = unnormalized_cost - twl.logprobs.sum().item()
            # :-1 to remove the EOS token
            new_packed_node = packed_node.extend(twl.token_ids[:-1].tolist())

            result.append(
                FullSearchNode(
                    new_packed_node,
                    # TODO: Remove need to set partial_parse when finished
                    partial_parse=NullPartialParse(),
                    hidden_state=None,
                    is_finished=True,
                    # + 1 to account for the EOS token
                    cost=gnmt_length_normalization(
                        self.length_normalization,
                        new_unnorm_cost,
                        len(new_packed_node.tokens) + 1,
                    ),
                    unnormalized_cost=new_unnorm_cost,
                )
            )

        for twl, new_hidden_state, partial_parse in incomplete:
            new_unnorm_cost = unnormalized_cost - twl.logprobs.sum().item()
            new_packed_node = packed_node.extend(twl.token_ids.tolist())

            result.append(
                FullSearchNode(
                    new_packed_node,
                    partial_parse,
                    hidden_state=new_hidden_state,
                    cost=gnmt_length_normalization(
                        self.length_normalization,
                        new_unnorm_cost,
                        len(new_packed_node.tokens),
                    ),
                    unnormalized_cost=new_unnorm_cost,
                )
            )

        if not result:
            # Speculative decoding failed completely, so back off to the regular method
            return await self.constrained_decoding_problem.expand(maybe_packed_node)

        return result
