# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Implementations of FSAs (finite state automata).

Currently contains an implementation of NFAs (nondeterministic finite automata).

This is useful for allowing expansions in grammars that are regular expressions.
"""

# TODO:
# For having a giant FSA which models the entire grammar (https://openreview.net/pdf?id=pK5HUWE-mz5):
# - We will have the rule "X -> A B" turned into the linear FST:
#        A     B     X̂
#     0 --> 1 --> 2 --> [3]
#   where the numbers are states and [3] is a final state.
#
#   We will need to know whether a path exists from a state which is labeled with X̂.
#   In the above example, we want to know that the 0, 1, and 2 states eventually lead to X̂.
#                                  *Â
#   In the OpenReview monograph: q ⟿ q'
#
# For disallowing nonterminals from producing certain strings:
# - Construct FSTs including node IDs and then put them inside a different FST later
#   We want to be able to use https://www.openfst.org/twiki/bin/view/FST/DifferenceDoc for this.

import bisect
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from openfst_python import Arc, MutableFst, VectorFst, Weight, determinize

from semantic_parsing_with_constrained_lm.util.unit import UNIT, Unit

# I for "Input"
# NFAStates transition to other NFAStates using Inputs (and other constituents of Edge, below).
# This should not be set to numpy.uint8 or semantic_parsing_with_constrained_lm.util.unit.Unit.
I = TypeVar("I", contravariant=True)


EPS = UNIT

# Things that can label an edge:
# - an Input. We will use this for nonterminals and other arbitrary grammar items.
# - a uint8, between 1 and 255 (0 is banned because OpenFST uses that for
#   epsilon). We will use this for UTF-8 bytes.
# - epsilon (Unit)
Edge = Union[I, np.uint8, Unit]


@dataclass
class BytesAndLabelsIndexer(Generic[I]):
    _edge_to_id: Dict[I, int] = dataclasses.field(default_factory=dict)
    _id_to_edge: List[I] = dataclasses.field(default_factory=list)

    def edge_to_id(self, edge: Edge[I]) -> int:
        if isinstance(edge, Unit):
            return 0
        if isinstance(edge, np.uint8):  # type: ignore
            if edge == 0:
                raise KeyError
            return int(edge)

        existing_id = self._edge_to_id.get(edge)
        if existing_id is None:
            new_id = len(self._id_to_edge) + 256
            self._id_to_edge.append(edge)
            self._edge_to_id[edge] = new_id
            return new_id
        else:
            return existing_id

    def id_to_edge(self, id_: int) -> Edge[I]:
        if id_ == 0:
            return EPS
        elif id_ < 256:
            return np.uint8(id_)
        return self._id_to_edge[id_ - 256]

    def num_ids(self) -> int:
        return len(self._id_to_edge) + 256

    def num_indexed(self) -> int:
        return len(self._id_to_edge)

    def freeze(self) -> "FrozenBytesAndLabelsIndexer[I]":
        return FrozenBytesAndLabelsIndexer(
            dict(self._edge_to_id), tuple(self._id_to_edge)
        )


@dataclass
class FrozenBytesAndLabelsIndexer(Generic[I]):
    """Like BytesAndLabelsIndexer, but disallows modifications."""

    _edge_to_id: Dict[I, int]
    _id_to_edge: Tuple[I, ...]

    def edge_to_id(self, edge: Edge[I]) -> Optional[int]:
        if isinstance(edge, Unit):
            return 0
        if isinstance(edge, np.uint8):  # type: ignore
            if edge == 0:
                raise KeyError
            return int(edge)
        return self._edge_to_id.get(edge)

    def id_to_edge(self, id_: int) -> Edge[I]:
        if id_ == 0:
            return EPS
        elif id_ < 256:
            return np.uint8(id_)
        return self._id_to_edge[id_ - 256]

    def num_ids(self) -> int:
        return len(self._id_to_edge) + 256

    def num_indexed(self) -> int:
        return len(self._id_to_edge)


@dataclass
class CompiledNFA(Generic[I]):
    """NFAStates translated into an OpenFST FST.

    Keeps track of integer IDs for edge labels which are not np.uint8.
    """

    fst: MutableFst
    edge_indexer: FrozenBytesAndLabelsIndexer[I]

    _zero_weight: Weight = dataclasses.field(init=False)

    def __post_init__(self):
        self._zero_weight = Weight.zero(self.fst.weight_type())

    def accepts(self, es: Iterable[Union[I, np.ubyte]]) -> bool:
        s = self._eps_closure({self.fst.start()})
        for e in es:
            s = self.transition_nfa(s, e)
            if not s:
                break
        return self.is_final_nfa(s)

    def accepts_str(self, es: str) -> bool:
        return self.accepts(np.frombuffer(es.encode("utf-8"), dtype=np.uint8))

    def _eps_closure(self, s: Set[int]) -> Set[int]:
        """Returns the set of all states reachable from `s` by epsilon transitions.

        The result is a superset of `s`."""
        frontier = s
        result: Set[int] = set(s)
        converged = False
        while not converged:
            converged = True
            new_frontier = set()
            for state in frontier:
                for arc in self.fst.arcs(state):
                    if arc.ilabel == 0 and arc.nextstate not in result:
                        result.add(arc.nextstate)
                        new_frontier.add(arc.nextstate)
                        converged = False
            frontier = new_frontier

        return result

    def is_final_nfa(self, s: Set[int]) -> bool:
        """Checks whether we have reached a final (accepting) state."""
        return len(s) > 0 and any(
            self.fst.final(state) != self._zero_weight for state in self._eps_closure(s)
        )

    def transition_nfa(self, s: Set[int], e: Union[I, np.ubyte]) -> Set[int]:
        edge_id = self.edge_indexer.edge_to_id(e)
        return {
            arc.nextstate
            for state in self._eps_closure(s)
            for arc in self.fst.arcs(state)
            if arc.ilabel == edge_id
        }


@dataclass
class CompiledDFA(CompiledNFA[I]):
    """Like CompiledNFA but all transitions are deterministic.

    This means:
    - No epsilon transitions
    - If a edge has a given label, there are no other edges from the
      state with the same label
    """

    # TODO: Maybe we can use matrix multiplications to do transitions in bulk?
    transition_array: np.ndarray = dataclasses.field(init=False)
    is_final_array: np.ndarray = dataclasses.field(init=False)

    # The outer list contains one element per node.
    # The inner lists contain the outgoing edge labels from the corresponding node.
    transition_labels: List[List[Union[I, np.uint8]]] = dataclasses.field(init=False)
    # The outer list contains one element per node.
    # The inner dicts contain the incoming edge labels from the corresponding node, in the keys;
    # the values are unused.
    # We use a dict with None values instead of a set because dicts remember insertion order.
    incoming_labels: List[Dict[Union[I, np.uint8], None]] = dataclasses.field(
        init=False
    )

    def __post_init__(self):
        super().__post_init__()

        self.transition_array = np.full(
            (self.fst.num_states(), self.edge_indexer.num_ids()), -1, dtype=np.int32
        )
        self.is_final_array = np.zeros((self.fst.num_states(),), dtype=bool)
        self.transition_labels = []
        self.incoming_labels = [{} for _ in range(self.fst.num_states())]

        for s in range(self.fst.num_states()):
            labels = []
            for arc in self.fst.arcs(s):
                self.transition_array[s, arc.ilabel] = arc.nextstate
                label = self.edge_indexer.id_to_edge(arc.ilabel)
                labels.append(label)
                self.incoming_labels[arc.nextstate][label] = None
            self.is_final_array[s] = self.fst.final(s) != self._zero_weight
            self.transition_labels.append(labels)

    @property
    def start_id(self) -> int:
        return self.fst.start()

    @staticmethod
    def from_nfa(nfa: CompiledNFA[I]) -> "CompiledDFA[I]":
        # Applies rmepsilon, determinize, minimize to the original FST
        fst = nfa.fst.copy()
        return CompiledDFA(determinize(fst.rmepsilon()).minimize(), nfa.edge_indexer)

    def accepts(self, es: Iterable[Union[I, np.ubyte]]) -> bool:
        s = self.fst.start()
        for e in es:
            s = self.transition_dfa(s, e)
            if s is None:
                return False
        return self.is_final_dfa(s)

    def is_final_dfa(self, s: int) -> bool:
        """Checks whether we have reached a final (accepting) state."""
        return self.is_final_array[s]

    def transition_dfa(self, s: int, e: Union[I, np.ubyte]) -> Optional[int]:
        edge_id = self.edge_indexer.edge_to_id(e)
        if edge_id is None:
            return None
        next_state = self.transition_array[s, edge_id]
        return None if next_state == -1 else next_state

    # Use reference equality
    def __eq__(self, other: Any) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


@dataclass
class CompiledNFABuilder(Generic[I]):
    """Used for building a CompiledNFA from NFAStates."""

    fst: MutableFst = dataclasses.field(default_factory=VectorFst)

    indexed_states: "Dict[NFAState[I], int]" = dataclasses.field(default_factory=dict)
    edge_indexer: BytesAndLabelsIndexer[I] = dataclasses.field(
        default_factory=BytesAndLabelsIndexer
    )

    def get_state_id(self, state: "NFAState[I]") -> Tuple[bool, int]:
        existing_id = self.indexed_states.get(state)
        if existing_id is None:
            new_id = self.fst.add_state()
            self.indexed_states[state] = new_id
            return False, new_id
        return True, existing_id

    def arc(self, edge: Edge[I], next_state_id: int) -> Arc:
        edge_id = self.edge_indexer.edge_to_id(edge)
        return Arc(
            ilabel=edge_id,
            olabel=edge_id,
            weight=Weight.one(self.fst.weight_type()),
            nextstate=next_state_id,
        )

    def build(self) -> CompiledNFA:
        return CompiledNFA(self.fst, self.edge_indexer.freeze())

    @staticmethod
    def compile(state: "NFAState[I]") -> CompiledNFA[I]:
        builder = CompiledNFABuilder[I]()
        state.compile(builder)
        return builder.build()


@dataclass(eq=False)
class NFAState(Generic[I], ABC):
    is_final: bool

    @abstractmethod
    def transition(self, e: Edge[I]) -> "Iterable[NFAState[I]]":
        """Returns the set of states reachable from this state by the given edge.

        TODO: Delete this since OpenFST subsumes it?"""
        pass

    def compile(self, builder: CompiledNFABuilder[I]) -> int:
        """Compiles this NFA state and its descendants into an OpenFST FST.

        Handles reentrancies and loops. Returns the ID of the state in the FST."""

        already_compiled, this_state_id = builder.get_state_id(self)
        if not already_compiled:
            self._compile(this_state_id, builder)
            builder.fst.set_start(this_state_id)
            if self.is_final:
                builder.fst.set_final(this_state_id)
        return this_state_id

    @abstractmethod
    def _compile(self, this_state_id: int, builder: CompiledNFABuilder[I]) -> None:
        """Child classes should implement this method for compilation."""
        pass


@dataclass(eq=False)
class Alternation(NFAState[I]):
    """FSA state where all outgoing transitions are labeled with epsilon.

    Used to implement alternation in regular expressions: `A|B`.
    For that, we need to create a state that has two outgoing epsilon transitions
    to states that can accept A and B."""

    next: Tuple[NFAState[I], ...]

    def transition(self, e: Edge[I]) -> Iterable[NFAState[I]]:
        if isinstance(e, Unit):
            return self.next
        return ()

    def _compile(self, this_state_id: int, builder: CompiledNFABuilder[I]) -> None:
        for next_node in self.next:
            next_state_id = next_node.compile(builder)
            builder.fst.add_arc(this_state_id, builder.arc(EPS, next_state_id))


@dataclass(eq=False)
class SingleInput(NFAState[I]):
    """Used for literal characters."""

    edge: Union[I, np.uint8]
    next: NFAState[I]

    def transition(self, e: Edge[I]) -> Iterable[NFAState[I]]:
        if e == self.edge:
            return (self.next,)
        return ()

    def _compile(self, this_state_id: int, builder: CompiledNFABuilder[I]) -> None:
        next_state_id = self.next.compile(builder)
        builder.fst.add_arc(this_state_id, builder.arc(self.edge, next_state_id))


@dataclass(eq=False)
class Sink(NFAState[I]):
    """Used for accepting states."""

    is_final: bool = True

    def transition(self, e: Edge[I]) -> Iterable[NFAState[I]]:
        return ()

    def _compile(self, this_state_id: int, builder: CompiledNFABuilder[I]) -> None:
        pass


@dataclass(eq=False)
class Ranges(NFAState[I]):
    """Compactly represents contiguous ranges of edges transitioning to the same state.

    This class doesn't support compilation to OpenFST, because we don't have a
    way to enumerate all edge values within a range."""

    # Must be sorted!
    bounds: Tuple[I, ...]
    # len(values) == len(bounds) + 1
    values: Tuple[Optional[NFAState[I]], ...]

    # Only if `applicable` returns true, do we look in `bounds`.
    # Otherwise, we don't allow a transition.
    applicable: Callable[[I], bool]

    def __post_init__(self):
        # ignore type because we assume that these bounds elements are comparable
        # TODO: Use type annotations to specify that, so we don't need the type: ignore.
        assert all(x <= y for x, y in zip(self.bounds, self.bounds[1:]))  # type: ignore
        assert len(self.bounds) + 1 == len(self.values)

    def transition(self, e: Union[I, Unit]) -> Iterable[NFAState[I]]:
        if not self.applicable(e) or isinstance(e, Unit):
            return ()

        # ignore type because we assume that these bounds elements, and e, are comparable
        next_node = self.values[bisect.bisect(self.bounds, e)]  # type: ignore
        return () if next_node is None else (next_node,)

    def _compile(self, this_state_id: int, builder: CompiledNFABuilder[I]) -> None:
        raise NotImplementedError


@dataclass(eq=False)
class UInt8Ranges(NFAState[I]):
    """Represents ranges of uint8 edges transitioning to the same state.

    Useful for implementing character ranges in UTF-8 encoded strings."""

    # Must be sorted!
    # We use int rather than uint8 because we would like to allow 256.
    bounds: Tuple[int, ...]
    # len(values) == len(bounds) + 1
    values: Tuple[Optional[NFAState[I]], ...]

    def __post_init__(self):
        assert all(0 <= x <= 256 for x in self.bounds)
        assert all(x <= y for x, y in zip(self.bounds, self.bounds[1:]))
        assert len(self.bounds) + 1 == len(self.values)

    def transition(self, e: Edge[I]) -> Iterable[NFAState[I]]:
        if not isinstance(e, np.uint8):  # type: ignore
            return ()

        next_node = self.values[bisect.bisect(self.bounds, int(e))]
        return () if next_node is None else (next_node,)

    def _compile(self, this_state_id: int, builder: CompiledNFABuilder[I]) -> None:
        for left, right, next_state in zip(
            (0,) + self.bounds, self.bounds + (256,), self.values
        ):
            if next_state is None:
                continue
            next_state_id = next_state.compile(builder)
            for i in range(left, right):
                builder.fst.add_arc(
                    this_state_id, builder.arc(np.uint8(i), next_state_id)
                )


#
# Below functions use NFAStates directly rather than going through OpenFST.
#
def eps_closure(s: Set[NFAState[I]]) -> Set[NFAState[I]]:
    """Returns the set of all states reachable from `s` by epsilon transitions.

    The result is a superset of `s`."""
    frontier = s
    result: Set[NFAState[I]] = set(s)
    converged = False
    while not converged:
        converged = True
        new_frontier = set()
        for state in frontier:
            for new_state in state.transition(UNIT):
                if new_state not in result:
                    result.add(new_state)
                    new_frontier.add(new_state)
                    converged = False
        frontier = new_frontier

    return result


def transition_nfa(s: Set[NFAState[I]], e: I) -> Set[NFAState[I]]:
    """Transition the NFA with the given input."""
    return {new_state for state in eps_closure(s) for new_state in state.transition(e)}


def is_final(s: Set[NFAState]) -> bool:
    """Checks whether we have reached a final (accepting) state."""
    return len(s) > 0 and any(state.is_final for state in eps_closure(s))


def accepts(s: Set[NFAState[I]], es: Iterable[I]) -> bool:
    """Checks whether the NFA accepts the given input."""
    s = eps_closure(s)
    for e in es:
        s = transition_nfa(s, e)
        if not s:
            break
    return is_final(s)
