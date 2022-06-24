# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Compile regular expression constructs into NFA nodes.

Currently supported:
- "|" (alternation)
- "?" (optional)
- "{m,n}" (between m and n repetitions, inclusive)
- "*" (0 or more repetitions)
- "+" (1 or more repetitions)
- "[a-z]" (character ranges)


See test_fsa_builders.py for some example uses.
"""

import itertools
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import more_itertools
import numpy as np

from semantic_parsing_with_constrained_lm.earley.fsa import (
    Alternation,
    CompiledDFA,
    CompiledNFA,
    CompiledNFABuilder,
    I,
    NFAState,
    Ranges,
    SingleInput,
    Sink,
    UInt8Ranges,
)
from semantic_parsing_with_constrained_lm.earley.utf8_ranges import Utf8Sequences
from semantic_parsing_with_constrained_lm.util.span import Span, SpanSet
from semantic_parsing_with_constrained_lm.util.util import identity

# A collection of NFA nodes (states) with an undefined "out" node.
#
# NFA nodes contain edges to the subsequent states that can be reached from them.
# In effect, we have to construct the NFA nodes right-to-left, with the sink
# having no outgoing transitions being constructed first.
# Using a closure allows us to defer building the concrete NFA nodes until the
# state they will transition to is known.
NFAFrag = Callable[[NFAState[I]], NFAState[I]]


def compile_nfa(regex: NFAFrag[I]) -> CompiledNFA[I]:
    return CompiledNFABuilder.compile(regex(Sink()))


def compile_dfa(regex: NFAFrag[I]) -> CompiledDFA[I]:
    return CompiledDFA.from_nfa(compile_nfa(regex))


def re_alternative(*frags: NFAFrag[I]) -> NFAFrag[I]:
    """Handles the | operator in regular expressions"""
    if len(frags) == 1:
        return frags[0]
    return lambda out: Alternation(
        is_final=False, next=tuple(frag(out) for frag in frags)
    )


def re_optional(frag: NFAFrag[I]) -> NFAFrag[I]:
    """Handles the ? operator"""
    return lambda out: Alternation(is_final=False, next=(frag(out), out))


def re_repetitions(
    frag: NFAFrag[I], minimum: int, maximum: Optional[int]
) -> NFAFrag[I]:
    """Handles the {m,n} operator (between m and n repetitions, inclusive)

    maximum (= n) can be None to indicate no upper bound."""

    def fragment(out: NFAState[I]) -> NFAState[I]:
        # Handle generating [minimum, maximum) repetitions
        if maximum is None:
            tail = Alternation(is_final=False, next=(out,))
            tail.next = tail.next + (frag(tail),)
            out = tail
        else:
            for _ in range(maximum - minimum):
                out = re_optional(frag)(out)

        # Prepend generating minimum repetitions, and chain the above after it
        for _ in range(minimum):
            out = frag(out)
        return out

    return fragment


def re_kleene_star(frag: NFAFrag[I]) -> NFAFrag[I]:
    """Handles the * operator (0 or more repetitions)"""
    return re_repetitions(frag, 0, None)


def re_plus(frag: NFAFrag[I]) -> NFAFrag[I]:
    """Handles the * operator (1 or more repetitions)"""
    return re_repetitions(frag, 1, None)


def re_atom(c: Union[I, np.uint8]) -> NFAFrag[I]:
    """Accepts a single character"""
    return lambda out: SingleInput(is_final=False, edge=c, next=out)


def re_sequence(cs: Iterable[Union[I, np.uint8]]) -> NFAFrag[I]:
    return re_concat(*(re_atom(c) for c in cs))


def re_utf8(s: str) -> NFAFrag[I]:
    return re_concat(
        *(re_atom(c) for c in np.frombuffer(s.encode("utf-8"), dtype=np.uint8))
    )


def re_concat(*frags: NFAFrag[I]) -> NFAFrag[I]:
    """Accepts the fragments in order"""
    if len(frags) == 0:
        return identity

    if len(frags) == 1:
        return frags[0]

    def fragment(out: NFAState[I]) -> NFAState[I]:
        for frag in reversed(frags):
            out = frag(out)
        return out

    return fragment


def re_ranges_deprecated(
    *all_ranges: Tuple[I, I],
    applicable: Callable[[I], bool] = lambda x: isinstance(x, (int, str)),
) -> NFAFrag[I]:
    """Handles things like [a-z].

    TODO: Coalesce ranges, e.g. [a-yb-z] is the same as [a-z].
    Currently, this function will give an incorrect result if there are such overlapping ranges.
    """
    # We have type: ignore as we assume all_ranges contains comparable elements
    # even though they're not explicitly annotated as such.

    assert all(x < y for x, y in all_ranges)  # type: ignore

    def fragment(out: NFAState[I]) -> NFAState[I]:
        bounds: Tuple[I, ...] = tuple(sorted(itertools.chain.from_iterable(all_ranges)))  # type: ignore
        values = tuple(itertools.islice(itertools.cycle((None, out)), len(bounds) + 1))
        return Ranges(
            is_final=False, bounds=bounds, values=values, applicable=applicable
        )

    return fragment


def re_ranges_unicode(
    included: Sequence[Tuple[str, str]], excluded: Sequence[Tuple[str, str]] = ()
) -> NFAFrag[I]:
    """Handles things like [a-z] or [[a-z]--[aoeui]].

    The arguments are sequences of closed intervals. included=[("a", "z")] means both a and z are included.
    The resulting NFA accepts UTF-8 byte sequences.
    """
    assert all(len(x) == 1 and len(y) == 1 for x, y in included)
    included_ints = [(ord(x), ord(y)) for x, y in included]

    assert all(len(x) == 1 and len(y) == 1 for x, y in excluded)
    excluded_ints = [(ord(x), ord(y)) for x, y in excluded]

    # SpanSet coalesces ranges for us, so that e.g. a-y and b-z are combined into a-z
    return re_ranges_span_set(
        SpanSet.new(Span.inclusive(x, y) for x, y in included_ints).difference(
            SpanSet.new(Span.inclusive(x, y) for x, y in excluded_ints)
        )
    )


def re_ranges_span_set(included_span_set: SpanSet) -> NFAFrag[I]:
    """Makes a NFA to accept UTF-8 bytes which correspond to ranges of code points.

    The diagrams at the top of https://swtch.com/~rsc/regexp/regexp3.html illustrate what these NFAs can look like."""

    assert included_span_set, "Must contain at least one span"
    assert (
        included_span_set.spans[-1].end <= 0x110000
    )  # Maximum Unicode character is 10FFFF

    def fragment(out: NFAState[I]) -> NFAState[I]:
        heads: List[UInt8Ranges] = []
        for span in included_span_set.spans:
            for seq in Utf8Sequences.from_span(span):
                last = out
                for range_ in seq.reverse().ranges:
                    bounds = (int(range_.start), int(range_.end + 1))
                    values = (None, last, None)
                    last = UInt8Ranges(is_final=False, bounds=bounds, values=values)
                assert isinstance(last, UInt8Ranges)
                heads.append(last)

        if len(heads) == 1:
            return heads[0]
        else:
            return Alternation(is_final=False, next=tuple(heads))

    return fragment


def re_substring_utf8(s: str, empty_allowed: bool = True) -> NFAFrag[I]:
    """Matches any substring of `s.encode("utf-8")` which is also valid UTF-8.

    For example, ❄ is \xe2\x9d\x84 and 👻 is \xf0\x9f\x91\xbb.
    If s is "❄👻", the resulting NFA would accept:
    - empty string (if empty_allowed is True)
    - \xe2\x9d\x84
    - \xf0\x9f\x91\xbb
    - \xe2\x9d\x84\xf0\x9f\x91\xbb
    but not something else like
    - \xe2
    - \xe2\x9d
    - \xf0\x9f\x91
    - \x84\xf0
    since those are not valid UTF-8 even if they are substrings of the encoded byte sequence.
    """

    encoded = s.encode("utf-8")
    # Every code point in an UTF-8 encoded string starts with a byte of the form 0xxxxxxx or 11xxxxxx.
    # This gives us sequences of bytes which correspond to a code point.
    code_point_chunks = list(
        more_itertools.split_before(encoded, lambda x: x < 0b10000000 or x > 0b11000000)
    )

    def fragment(end: NFAState[I]) -> NFAState[I]:
        code_point_starts: List[NFAState[I]] = []
        out = end
        for code_point_chunk in reversed(code_point_chunks):
            accepts_this_chunk = re_sequence(np.uint8(x) for x in code_point_chunk)(out)
            out = Alternation(
                is_final=False,
                next=(accepts_this_chunk, end),
            )
            if empty_allowed:
                code_point_starts.append(out)
            else:
                code_point_starts.append(accepts_this_chunk)
        return Alternation(is_final=False, next=tuple(code_point_starts))

    return fragment
