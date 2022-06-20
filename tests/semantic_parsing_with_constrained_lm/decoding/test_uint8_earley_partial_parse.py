# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=redefined-outer-name
# This file is modeled after test_earley_partial_parse.py
from typing import List, TypeVar, Union

import numpy as np
import pytest
import torch

from semantic_parsing_with_constrained_lm.earley.cfg import load_grammar_from_string
from semantic_parsing_with_constrained_lm.earley.grammar import Grammar
from semantic_parsing_with_constrained_lm.util.unit import Unit
from semantic_parsing_with_constrained_lm.decoding.partial_parse import PartialParse
from semantic_parsing_with_constrained_lm.decoding.uint8_earley_partial_parse import (
    UInt8EarleyPartialParse,
    UInt8GrammarTokenizerInfo,
)

# This grammar can generate:
#   abcA
#   abcB
#   abcC
#   abcDE
SIMPLE_GRAMMAR = """
start -> a
start -> b
start -> c
a -> "a" "bcA"
a -> "ab" "cB"
b -> "abc" "C"
b -> c "DE"
c -> "a" "bc"
"""


@pytest.fixture(scope="module")
def simple_grammar() -> Grammar[np.uint8, Unit]:
    return load_grammar_from_string(SIMPLE_GRAMMAR)


S = TypeVar("S", str, bytes)


def make_helpers(vocab: List[S]):
    token_to_id = {token: i for i, token in enumerate(vocab)}

    def compare(
        partial_parse: PartialParse, expected_tokens: List[S], expected_can_end: bool
    ) -> None:
        tokens, can_end = partial_parse.allowed_next(
            torch.arange(len(vocab), dtype=torch.long)
        )
        assert tokens is not None
        assert set(tokens.tolist()) == set(token_to_id[t] for t in expected_tokens)
        assert can_end == expected_can_end

    def append(partial_parse: PartialParse, token: S) -> PartialParse:
        return partial_parse.append(token_to_id[token])

    return compare, append


def initial_factory(
    vocab: Union[List[str], List[bytes]], grammar: Grammar[np.uint8, Unit]
) -> PartialParse:
    encoded_vocab = [
        np.frombuffer(v.encode("utf-8") if isinstance(v, str) else v, dtype=np.uint8)
        for v in vocab
    ]
    info = UInt8GrammarTokenizerInfo(grammar, encoded_vocab)
    return UInt8EarleyPartialParse.initial(info)


def test_simple_grammar_single_char_vocab(
    simple_grammar: Grammar[np.uint8, Unit]
) -> None:
    single_char_vocab = ["a", "b", "c", "A", "B", "C", "D", "E"]
    initial = initial_factory(single_char_vocab, simple_grammar)
    compare, append = make_helpers(single_char_vocab)

    compare(initial, ["a"], False)
    pp_a = append(initial, "a")
    compare(pp_a, ["b"], False)
    pp_ab = append(pp_a, "b")
    compare(pp_ab, ["c"], False)
    pp_abc = append(pp_ab, "c")
    compare(pp_abc, ["A", "B", "C", "D"], True)
    pp_abcA = append(pp_abc, "A")
    compare(pp_abcA, [], True)
    pp_abcB = append(pp_abc, "B")
    compare(pp_abcB, [], True)
    pp_abcC = append(pp_abc, "C")
    compare(pp_abcC, [], True)
    pp_abcD = append(pp_abc, "D")
    compare(pp_abcD, ["E"], False)
    pp_abcDE = append(pp_abcD, "E")
    compare(pp_abcDE, [], True)


def test_simple_grammar_multi_char_vocab(
    simple_grammar: Grammar[np.uint8, Unit]
) -> None:
    multi_char_vocab = [
        "a",
        "b",
        "c",
        "ab",
        "bc",
        "cD",
        "A",
        "B",
        "C",
        "D",
        "E",
        "DE",
        "abcD",
        "abcdef",
    ]
    initial = initial_factory(multi_char_vocab, simple_grammar)
    compare, append = make_helpers(multi_char_vocab)

    compare(initial, ["a", "ab", "abcD"], False)
    pp_a = append(initial, "a")
    compare(pp_a, ["b", "bc"], False)
    pp_ab_1 = append(initial, "ab")
    compare(pp_ab_1, ["c", "cD"], False)
    pp_ab_2 = append(pp_a, "b")
    compare(pp_ab_2, ["c", "cD"], False)
    pp_abc_1 = append(pp_ab_1, "c")
    compare(pp_abc_1, ["A", "B", "C", "D", "DE"], True)
    pp_abc_2 = append(pp_a, "bc")
    compare(pp_abc_2, ["A", "B", "C", "D", "DE"], True)
    pp_abcA_1 = append(pp_abc_1, "A")
    compare(pp_abcA_1, [], True)
    pp_abcA_2 = append(pp_abc_2, "A")
    compare(pp_abcA_2, [], True)
    pp_abcD_1 = append(initial, "abcD")
    compare(pp_abcD_1, ["E"], False)
    pp_abcD_2 = append(pp_abc_1, "D")
    compare(pp_abcD_2, ["E"], False)
    pp_abcDE_1 = append(pp_abcD_1, "E")
    compare(pp_abcDE_1, [], True)
    pp_abcDE_2 = append(pp_abcD_2, "E")
    compare(pp_abcDE_2, [], True)

    # Try again but only with one path
    initial = initial_factory(multi_char_vocab, simple_grammar)
    compare(initial, ["a", "ab", "abcD"], False)
    pp_ab = append(initial, "ab")
    compare(pp_ab, ["c", "cD"], False)
    pp_abcD = append(pp_ab, "cD")
    compare(pp_abcD, ["E"], False)
    pp_abcDE = append(pp_abcD, "E")
    compare(pp_abcDE, [], True)


EMOJI_GRAMMAR = """
start -> cold monster
cold ->    "\N{snowman}"
cold ->    "\N{snowflake}"
cold ->    "\N{freezing face}"
monster -> "\N{ghost}"
monster -> "\N{alien monster}"
monster -> "\N{biohazard sign}"
"""
# UTF-8 encoded:
# snowman: b'\xe2\x98\x83'
# snowflake: b'\xe2\x9d\x84'
# freezing face: b'\xf0\x9f\xa5\xb6'
# ghost: b'\xf0\x9f\x91\xbb'
# alien monster: b'\xf0\x9f\x91\xbe'
# biohazard sign: b'\xe2\x98\xa3'


@pytest.fixture(scope="module")
def emoji_grammar() -> Grammar[np.uint8, Unit]:
    return load_grammar_from_string(EMOJI_GRAMMAR)


def test_emoji_grammar_utf8(emoji_grammar: Grammar[np.uint8, Unit]) -> None:
    vocab = [
        "\N{snowman}".encode("utf-8"),
        "\N{snowflake}".encode("utf-8"),
        "\N{freezing face}".encode("utf-8"),
        "\N{ghost}".encode("utf-8"),
        "\N{alien monster}".encode("utf-8"),
        "\N{biohazard sign}".encode("utf-8"),
        b"\x98",
        b"\x9d",
        b"\xe2",
        b"\xf0",
        b"\x83\xf0",
        b"\x98\x83",
        b"\x9f\x91",
        b"\xe2\x98",
        b"\x9f\xa5\xb6",
        b"\xf0\x9f\x91",
    ]
    initial = initial_factory(vocab, emoji_grammar)
    compare, append = make_helpers(vocab)

    compare(
        initial,
        [
            "\N{snowman}".encode("utf-8"),
            "\N{snowflake}".encode("utf-8"),
            "\N{freezing face}".encode("utf-8"),
            b"\xe2",
            b"\xe2\x98",
            b"\xf0",
        ],
        False,
    )
    pp_snowman = append(initial, "\N{snowman}".encode("utf-8"))
    after_snowman = [
        "\N{ghost}".encode("utf-8"),
        "\N{alien monster}".encode("utf-8"),
        "\N{biohazard sign}".encode("utf-8"),
        b"\xe2",
        b"\xe2\x98",
        b"\xf0",
        b"\xf0\x9f\x91",
    ]
    compare(pp_snowman, after_snowman, False)
    pp_xf0 = append(initial, b"\xf0")
    compare(pp_xf0, [b"\x9f\xa5\xb6"], False)
    pp_snowman_ghost = append(pp_snowman, "\N{ghost}".encode("utf-8"))
    compare(pp_snowman_ghost, [], True)
    pp_xe2 = append(initial, b"\xe2")
    compare(pp_xe2, [b"\x98", b"\x9d", b"\x98\x83"], False)
    pp_xe2_x98 = append(pp_xe2, b"\x98")
    compare(pp_xe2_x98, [b"\x83\xf0"], False)
    pp_xe2_x98_x83 = append(pp_xe2, b"\x98\x83")
    compare(pp_xe2_x98_x83, after_snowman, False)
    pp_xe2_x98_x83_xf0 = append(pp_xe2_x98, b"\x83\xf0")
    compare(pp_xe2_x98_x83_xf0, [b"\x9f\x91"], False)


PARENS_GRAMMAR = """
start -> round | square
inner -> start | #e
round -> "(" inner ")" | "((" inner "))"
square -> "[" inner "]" | "[[[" inner "]]]"
"""


@pytest.fixture(scope="module")
def parens_grammar() -> Grammar[np.uint8, Unit]:
    return load_grammar_from_string(PARENS_GRAMMAR)


def test_parens_grammar(parens_grammar: Grammar[np.uint8, Unit]) -> None:
    vocab1 = ["(", "[", ")", "]"]
    initial = initial_factory(vocab1, parens_grammar)
    compare, append = make_helpers(vocab1)

    compare(initial, ["(", "["], False)
    pp_r = append(initial, "(")
    compare(pp_r, ["(", "[", ")"], False)
    pp_rR = append(pp_r, ")")
    compare(pp_rR, [], True)
    pp_rs = append(pp_r, "[")
    compare(pp_rs, ["(", "[", "]"], False)
    pp_rsS = append(pp_rs, "]")
    compare(pp_rsS, [")"], False)
    pp_sss = append(append(append(initial, "["), "["), "[")
    compare(pp_sss, ["(", "[", "]"], False)

    vocab2 = ["((", ")))"]
    initial = initial_factory(vocab2, parens_grammar)
    compare, append = make_helpers(vocab2)

    compare(initial, ["(("], False)
    pp_r2 = append(initial, "((")
    compare(pp_r2, ["(("], False)
    pp_r4 = append(pp_r2, "((")
    compare(pp_r4, ["((", ")))"], False)
    pp_r4R3 = append(pp_r4, ")))")
    compare(pp_r4R3, [], False)
    pp_r6 = append(pp_r4, "((")
    compare(pp_r6, ["((", ")))"], False)
    pp_r6R3 = append(pp_r6, ")))")
    compare(pp_r6R3, [")))"], False)
    pp_r6R6 = append(pp_r6R3, ")))")
    compare(pp_r6R6, [], True)
