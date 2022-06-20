# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path

import numpy as np

from semantic_parsing_with_constrained_lm.earley.cfg import load_grammar_from_directory, parse_re_char_set
from semantic_parsing_with_constrained_lm.earley.recognize import is_grammatical
from semantic_parsing_with_constrained_lm.util.span import Span, SpanSet


def test_parse_re_char_set() -> None:
    assert parse_re_char_set("") == SpanSet()
    assert parse_re_char_set("a") == SpanSet.new([Span.inclusive(ord("a"), ord("a"))])
    az = parse_re_char_set("a-z")
    assert ord("a") in az
    assert ord("z") in az
    assert ord("a") - 1 not in az
    assert ord("z") + 1 not in az
    assert ord("A") not in az

    assert parse_re_char_set(r"\--\\") == SpanSet.new(
        [Span.inclusive(ord("-"), ord("\\"))]
    )
    assert parse_re_char_set(r"\--\\\u1234") == SpanSet.new(
        [Span.inclusive(ord("-"), ord("\\")), Span.inclusive(0x1234, 0x1234)]
    )


def test_parsing() -> None:
    def parses(s: str) -> bool:
        nonlocal grammar
        return is_grammatical(
            list(np.frombuffer(s.encode("utf-8"), dtype=np.uint8)), grammar
        )

    grammar_path = str(Path(__file__).parent / "test_grammar")

    # r1
    grammar = load_grammar_from_directory(grammar_path, start_nt="r1")
    assert not parses("")
    assert parses("a")
    assert not parses("b")
    assert parses("abbab")
    assert parses("abbaabaa")

    # r2
    grammar = load_grammar_from_directory(grammar_path, start_nt="r2")
    assert not parses("c")
    assert not parses("d")
    assert parses("cc")
    assert parses("cd")
    assert parses("dc")
    assert parses("dd")
    assert not parses("ccc")

    # r3
    grammar = load_grammar_from_directory(grammar_path, start_nt="r3")
    assert parses("")
    assert parses("e")
    assert parses("eee")
    assert parses("efefe")
    assert not parses("effe")
    assert not parses("ffe")
    assert not parses("eeeee")

    # r4
    grammar = load_grammar_from_directory(grammar_path, start_nt="r4")
    assert not parses("")
    assert not parses("gh")
    assert parses("ghh")
    assert parses("ghgghh")

    # r5
    grammar = load_grammar_from_directory(grammar_path, start_nt="r5")
    assert not parses("")
    assert not parses("i")
    assert parses("iiiii")
    assert parses("iiiij")
    assert parses("iiiijj")
    assert parses("iiiijjj")
    assert not parses("iiiijjjj")
    assert parses("ijjiiiii")
    assert not parses("ijjiiiiij")

    # r6
    grammar = load_grammar_from_directory(grammar_path, start_nt="r6")
    assert not parses("")
    assert parses("k")
    assert not parses("l")
    assert not parses("m")
    assert parses("n")
