# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Same as test_fsa_builders.py, but using the FST compilation

from typing import Callable

import more_itertools
import numpy as np
import pytest

from semantic_parsing_with_constrained_lm.earley.fsa import CompiledNFA
from semantic_parsing_with_constrained_lm.earley.fsa_builders import (
    NFAFrag,
    compile_dfa,
    compile_nfa,
    re_alternative,
    re_atom,
    re_concat,
    re_kleene_star,
    re_optional,
    re_plus,
    re_ranges_span_set,
    re_ranges_unicode,
    re_repetitions,
    re_sequence,
    re_substring_utf8,
)
from semantic_parsing_with_constrained_lm.util.span import Span, SpanSet


@pytest.mark.parametrize("_compile", [compile_nfa, compile_dfa])
def test_single_character(_compile: Callable[[NFAFrag[str]], CompiledNFA[str]]):
    exp = _compile(re_atom("x"))
    assert exp.accepts("x")
    assert not exp.accepts("")
    assert not exp.accepts("y")


@pytest.mark.parametrize("_compile", [compile_nfa, compile_dfa])
def test_single_string(_compile: Callable[[NFAFrag[str]], CompiledNFA[str]]):
    # "xyz"
    exp = _compile(re_sequence("xyz"))
    assert exp.accepts("xyz")
    assert not exp.accepts("x")
    assert not exp.accepts("xy")
    assert not exp.accepts("")


@pytest.mark.parametrize("_compile", [compile_nfa, compile_dfa])
def test_alternative(_compile: Callable[[NFAFrag[str]], CompiledNFA[str]]):
    # "x" | "yz"
    exp = _compile(re_alternative(re_atom("x"), re_sequence("yz")))
    assert exp.accepts("x")
    assert exp.accepts("yz")
    assert not exp.accepts("")
    assert not exp.accepts("z")
    assert not exp.accepts("y")


@pytest.mark.parametrize("_compile", [compile_nfa, compile_dfa])
def test_optional(_compile: Callable[[NFAFrag[str]], CompiledNFA[str]]):
    # "x?"
    exp = _compile(re_optional(re_atom("x")))
    assert exp.accepts("x")
    assert exp.accepts("")
    assert not exp.accepts("y")

    # "x?y"
    exp = _compile(re_concat(re_optional(re_atom("x")), re_atom("y")))
    assert exp.accepts("xy")
    assert exp.accepts("y")
    assert not exp.accepts("x")
    assert not exp.accepts("")

    # "x(yz)?"
    exp = _compile(re_concat(re_atom("x"), re_optional(re_sequence("yz"))))
    assert exp.accepts("xyz")
    assert not exp.accepts("xy")
    assert exp.accepts("x")
    assert not exp.accepts("")


@pytest.mark.parametrize("_compile", [compile_nfa, compile_dfa])
def test_kleene_star(_compile: Callable[[NFAFrag[str]], CompiledNFA[str]]):
    # "x*"
    exp = _compile(re_kleene_star(re_atom("x")))
    assert exp.accepts("x")
    assert exp.accepts("xx")
    assert exp.accepts("xxxxx")
    assert exp.accepts("")
    assert not exp.accepts("y")

    # "(xy)*"
    exp = _compile(re_kleene_star(re_sequence("xy")))
    assert exp.accepts("xy")
    assert not exp.accepts("xyx")
    assert exp.accepts("xyxy")
    assert exp.accepts("")


@pytest.mark.parametrize("_compile", [compile_nfa, compile_dfa])
def test_plus(_compile: Callable[[NFAFrag[str]], CompiledNFA[str]]):
    # "x+"
    exp = _compile(re_plus(re_atom("x")))
    assert exp.accepts("x")
    assert exp.accepts("xx")
    assert exp.accepts("xxxxx")
    assert not exp.accepts("")
    assert not exp.accepts("y")

    # "(xy)+"
    exp = _compile(re_plus(re_sequence("xy")))
    assert exp.accepts("xy")
    assert not exp.accepts("xyx")
    assert exp.accepts("xyxy")
    assert not exp.accepts("")


@pytest.mark.parametrize("_compile", [compile_nfa, compile_dfa])
def test_repetitions(_compile: Callable[[NFAFrag[str]], CompiledNFA[str]]):
    # "x{2}"
    exp = _compile(re_repetitions(re_atom("x"), 2, 2))
    assert not exp.accepts("x")
    assert exp.accepts("xx")
    assert not exp.accepts("xxxxx")
    assert not exp.accepts("")
    assert not exp.accepts("y")

    # "x{2,}"
    exp = _compile(re_repetitions(re_atom("x"), 2, None))
    assert not exp.accepts("x")
    assert exp.accepts("xx")
    assert exp.accepts("xxx")
    assert exp.accepts("xxxxx")
    assert exp.accepts("xxxxxxxxx")
    assert not exp.accepts("")
    assert not exp.accepts("y")

    # "x{3,5}"
    exp = _compile(re_repetitions(re_atom("x"), 3, 5))
    assert not exp.accepts("x" * 0)
    assert not exp.accepts("x" * 1)
    assert not exp.accepts("x" * 2)
    assert exp.accepts("x" * 3)
    assert exp.accepts("x" * 4)
    assert exp.accepts("x" * 5)
    assert not exp.accepts("x" * 6)

    # "x{3,)y{3,}"
    exp = _compile(
        re_concat(
            re_repetitions(re_atom("x"), 3, None), re_repetitions(re_atom("y"), 3, None)
        )
    )
    for i in range(1, 5):
        for j in range(1, 5):
            s = "x" * i + "y" * j
            should_accept = i >= 3 and j >= 3
            assert exp.accepts(s) == should_accept


@pytest.mark.parametrize("_compile", [compile_nfa, compile_dfa])
def test_ranges(_compile: Callable[[NFAFrag[str]], CompiledNFA[str]]):
    # "[a-z]"
    exp = _compile(re_ranges_unicode([("a", "z")]))
    assert exp.accepts_str("a")
    assert exp.accepts_str("z")
    assert not exp.accepts_str("A")
    assert not exp.accepts_str("0")
    assert not exp.accepts_str("{")

    # "[a-zA-Z]"
    exp = _compile(re_ranges_unicode([("a", "z"), ("A", "Z")]))
    assert exp.accepts_str("a")
    assert exp.accepts_str("z")
    assert exp.accepts_str("A")
    assert not exp.accepts_str("0")
    assert not exp.accepts_str("{")
    assert not exp.accepts_str("[")

    # "[[a-zA-Z]--[X-b]]"
    exp = _compile(re_ranges_unicode([("a", "z"), ("A", "Z")], [("X", "b")]))
    assert not exp.accepts_str("a")
    assert not exp.accepts_str("b")
    assert exp.accepts_str("c")
    assert exp.accepts_str("z")
    assert exp.accepts_str("A")
    assert exp.accepts_str("W")
    assert not exp.accepts_str("X")
    assert not exp.accepts_str("0")
    assert not exp.accepts_str("{")
    assert not exp.accepts_str("[")

    # "[\u4000-\u502F]"
    exp = _compile(re_ranges_unicode([("\u4000", "\u502F")]))
    assert exp.accepts_str("\u4000")
    assert exp.accepts_str("\u502F")
    assert not exp.accepts_str("\u3FFF")
    assert not exp.accepts_str("\u5030")
    assert not exp.accepts_str("{")


@pytest.mark.parametrize("_compile", [compile_nfa, compile_dfa])
def test_ranges_span_set(_compile: Callable[[NFAFrag[str]], CompiledNFA[str]]):
    # "[a-z]"
    exp = _compile(
        re_ranges_span_set(SpanSet.new([Span.inclusive(ord("a"), ord("z"))]))
    )
    assert exp.accepts_str("a")
    assert exp.accepts_str("z")
    assert not exp.accepts_str("A")
    assert not exp.accepts_str("0")
    assert not exp.accepts_str("{")

    # "[a-zA-Z]"
    exp = _compile(
        re_ranges_span_set(
            SpanSet.new(
                [Span.inclusive(ord("a"), ord("z")), Span.inclusive(ord("A"), ord("Z"))]
            )
        )
    )
    assert exp.accepts_str("a")
    assert exp.accepts_str("z")
    assert exp.accepts_str("A")
    assert not exp.accepts_str("0")
    assert not exp.accepts_str("{")
    assert not exp.accepts_str("[")

    # "[\u4000-\u502F]"
    exp = _compile(
        re_ranges_span_set(SpanSet.new([Span.inclusive(ord("\u4000"), ord("\u502F"))]))
    )
    assert exp.accepts_str("\u4000")
    assert exp.accepts_str("\u502F")
    assert not exp.accepts_str("\u3FFF")
    assert not exp.accepts_str("\u5030")
    assert not exp.accepts_str("{")


@pytest.mark.parametrize("_compile", [compile_nfa, compile_dfa])
def test_substring(_compile: Callable[[NFAFrag[str]], CompiledNFA[str]]):
    def exhaustive(s: str) -> None:
        """Checks that re_substring_utf8 accepts only those valid substrings of `s`, among all subsets of `s`"""
        for empty_allowed in (True, False):
            exp = _compile(re_substring_utf8(s, empty_allowed))
            s_bytes = s.encode("utf-8")
            for subset in more_itertools.unique_everseen(
                more_itertools.powerset(s_bytes)
            ):
                subset_bytes = bytes(subset)
                is_substr = subset_bytes in s_bytes and (
                    len(subset_bytes) > 0 or empty_allowed
                )
                is_valid_utf8 = True
                try:
                    subset_bytes.decode("utf-8")
                except:
                    is_valid_utf8 = False
                assert exp.accepts(np.frombuffer(subset_bytes, dtype=np.uint8)) == (
                    is_substr and is_valid_utf8
                )

    exhaustive("aardvark")
    exhaustive("éclair")
    exhaustive("\N{snowflake}\N{ghost}")
    exhaustive("a\N{snowflake}b\N{ghost}c")
