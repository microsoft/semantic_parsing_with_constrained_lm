# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

from semantic_parsing_with_constrained_lm.earley.fsa import Sink, accepts
from semantic_parsing_with_constrained_lm.earley.fsa_builders import (
    re_alternative,
    re_atom,
    re_concat,
    re_kleene_star,
    re_optional,
    re_plus,
    re_ranges_deprecated,
    re_ranges_unicode,
    re_repetitions,
    re_sequence,
)


def test_single_character():
    # "x"
    exp = {re_atom("x")(Sink())}
    assert accepts(exp, "x")
    assert not accepts(exp, "")
    assert not accepts(exp, "y")


def test_single_string():
    # "xyz"
    exp = {re_sequence("xyz")(Sink())}
    assert accepts(exp, "xyz")
    assert not accepts(exp, "x")
    assert not accepts(exp, "xy")
    assert not accepts(exp, "")


def test_alternative():
    # "x" | "yz"
    exp = {re_alternative(re_atom("x"), re_sequence("yz"))(Sink())}
    assert accepts(exp, "x")
    assert accepts(exp, "yz")
    assert not accepts(exp, "")
    assert not accepts(exp, "z")
    assert not accepts(exp, "y")


def test_optional():
    # "x?"
    exp = {re_optional(re_atom("x"))(Sink())}
    assert accepts(exp, "x")
    assert accepts(exp, "")
    assert not accepts(exp, "y")

    # "x?y"
    exp = {re_concat(re_optional(re_atom("x")), re_atom("y"))(Sink())}
    assert accepts(exp, "xy")
    assert accepts(exp, "y")
    assert not accepts(exp, "x")
    assert not accepts(exp, "")

    # "x(yz)?"
    exp = {re_concat(re_atom("x"), re_optional(re_sequence("yz")))(Sink())}
    assert accepts(exp, "xyz")
    assert not accepts(exp, "xy")
    assert accepts(exp, "x")
    assert not accepts(exp, "")


def test_kleene_star():
    # "x*"
    exp = {re_kleene_star(re_atom("x"))(Sink())}
    assert accepts(exp, "x")
    assert accepts(exp, "xx")
    assert accepts(exp, "xxxxx")
    assert accepts(exp, "")
    assert not accepts(exp, "y")

    # "(xy)*"
    exp = {re_kleene_star(re_sequence("xy"))(Sink())}
    assert accepts(exp, "xy")
    assert not accepts(exp, "xyx")
    assert accepts(exp, "xyxy")
    assert accepts(exp, "")


def test_plus():
    # "x+"
    exp = {re_plus(re_atom("x"))(Sink())}
    assert accepts(exp, "x")
    assert accepts(exp, "xx")
    assert accepts(exp, "xxxxx")
    assert not accepts(exp, "")
    assert not accepts(exp, "y")

    # "(xy)+"
    exp = {re_plus(re_sequence("xy"))(Sink())}
    assert accepts(exp, "xy")
    assert not accepts(exp, "xyx")
    assert accepts(exp, "xyxy")
    assert not accepts(exp, "")


def test_repetitions():
    # "x{2}"
    exp = {re_repetitions(re_atom("x"), 2, 2)(Sink())}
    assert not accepts(exp, "x")
    assert accepts(exp, "xx")
    assert not accepts(exp, "xxxxx")
    assert not accepts(exp, "")
    assert not accepts(exp, "y")

    # "x{2,}"
    exp = {re_repetitions(re_atom("x"), 2, None)(Sink())}
    assert not accepts(exp, "x")
    assert accepts(exp, "xx")
    assert accepts(exp, "xxx")
    assert accepts(exp, "xxxxx")
    assert accepts(exp, "xxxxxxxxx")
    assert not accepts(exp, "")
    assert not accepts(exp, "y")

    # "x{3,5}"
    exp = {re_repetitions(re_atom("x"), 3, 5)(Sink())}
    assert not accepts(exp, "x" * 0)
    assert not accepts(exp, "x" * 1)
    assert not accepts(exp, "x" * 2)
    assert accepts(exp, "x" * 3)
    assert accepts(exp, "x" * 4)
    assert accepts(exp, "x" * 5)
    assert not accepts(exp, "x" * 6)

    # "x{3,)y{3,}"
    exp = {
        re_concat(
            re_repetitions(re_atom("x"), 3, None), re_repetitions(re_atom("y"), 3, None)
        )(Sink())
    }
    for i in range(1, 5):
        for j in range(1, 5):
            s = "x" * i + "y" * j
            should_accept = i >= 3 and j >= 3
            assert accepts(exp, s) == should_accept


def test_ranges():
    # "[a-z]"
    # str() so that pyright thinks the arguments are "str" rather than "Literal"
    # "{" because it is one after "z"
    exp = {re_ranges_deprecated((str("a"), "{"))(Sink())}
    assert accepts(exp, "a")
    assert accepts(exp, "z")
    assert not accepts(exp, "A")
    assert not accepts(exp, "0")
    assert not accepts(exp, "{")

    # "[a-zA-Z]"
    # "[" because it is one after "Z"
    exp = {re_ranges_deprecated((str("a"), "{"), ("A", "["))(Sink())}
    assert accepts(exp, "a")
    assert accepts(exp, "z")
    assert accepts(exp, "A")
    assert not accepts(exp, "0")
    assert not accepts(exp, "{")
    assert not accepts(exp, "[")


def test_ranges_unicode():
    # "[a-z]"
    exp = {re_ranges_unicode(included=[("a", "z")])(Sink())}
    assert not accepts(exp, np.frombuffer(b"`", dtype=np.uint8))
    assert accepts(exp, np.frombuffer(b"a", dtype=np.uint8))
    assert accepts(exp, np.frombuffer(b"z", dtype=np.uint8))
    assert not accepts(exp, np.frombuffer(b"{", dtype=np.uint8))

    # "[[a-z]--[aeiou]"
    exp = {
        re_ranges_unicode(
            included=[("a", "z")],
            excluded=[("a", "a"), ("e", "e"), ("i", "i"), ("o", "o"), ("u", "u")],
        )(Sink())
    }
    assert not accepts(exp, np.frombuffer(b"`", dtype=np.uint8))
    assert not accepts(exp, np.frombuffer(b"a", dtype=np.uint8))
    assert accepts(exp, np.frombuffer(b"b", dtype=np.uint8))
    assert accepts(exp, np.frombuffer(b"d", dtype=np.uint8))
    assert not accepts(exp, np.frombuffer(b"e", dtype=np.uint8))
    assert accepts(exp, np.frombuffer(b"f", dtype=np.uint8))
    assert not accepts(exp, np.frombuffer(b"u", dtype=np.uint8))
    assert accepts(exp, np.frombuffer(b"z", dtype=np.uint8))
    assert not accepts(exp, np.frombuffer(b"{", dtype=np.uint8))
