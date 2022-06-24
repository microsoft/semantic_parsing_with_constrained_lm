# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The code in this file is translated directly from the #[cfg(test)] section of
# https://github.com/rust-lang/regex/blob/258bdf798a14f50529c1665e84cc8a3a9e2c90fc/regex-syntax/src/utf8.rs

from typing import Sequence

import numpy as np

from semantic_parsing_with_constrained_lm.earley.utf8_ranges import Utf8Range, Utf8Sequence, Utf8Sequences


def rutf8(s: int, e: int) -> Utf8Range:
    assert 0 <= s <= 0xFF
    assert 0 <= e <= 0xFF
    return Utf8Range(np.uint8(s), np.uint8(e))


TAG_CONT = np.uint8(0b100000000)
TAG_THREE_B = np.uint8(0b11100000)


def encode_surrogate(cp: int) -> Sequence[np.uint8]:
    assert 0xD800 <= cp < 0xE000
    return (
        np.uint8(cp >> 12 & 0x0F) | TAG_THREE_B,
        np.uint8(cp >> 6 & 0x3F) | TAG_CONT,
        np.uint8(cp & 0x3F) | TAG_CONT,
    )


def never_accepts_surrogate_codepoints(start: str, end: str) -> None:
    for cp in range(0xD800, 0xE000):
        buf = encode_surrogate(cp)
        for r in Utf8Sequences.new(start, end):
            assert not r.matches(buf)


def test_codepoints_no_surrogates():
    never_accepts_surrogate_codepoints("\u0000", "\uFFFF")
    never_accepts_surrogate_codepoints("\u0000", "\U0010FFFF")
    never_accepts_surrogate_codepoints("\u0000", "\U0010FFFE")
    never_accepts_surrogate_codepoints("\u0080", "\U0010FFFF")
    never_accepts_surrogate_codepoints("\uD7FF", "\uE000")


def test_codepoint_one_sequence():
    """Tests that every range of scalar values that contains a single scalar
    value is recognized by one sequence of byte ranges.
    """
    for i in range(0x0, 0x10FFFF + 1):
        if 0xD800 <= i <= 0xDFFF:
            continue
        c = chr(i)
        seqs = list(Utf8Sequences.new(c, c))
        assert len(seqs) == 1, hex(i)


def test_bmp():
    seqs = list(Utf8Sequences.new("\u0000", "\uFFFF"))
    assert seqs == [
        Utf8Sequence([rutf8(0x0, 0x7F)]),
        Utf8Sequence([rutf8(0xC2, 0xDF), rutf8(0x80, 0xBF)]),
        Utf8Sequence([rutf8(0xE0, 0xE0), rutf8(0xA0, 0xBF), rutf8(0x80, 0xBF)]),
        Utf8Sequence([rutf8(0xE1, 0xEC), rutf8(0x80, 0xBF), rutf8(0x80, 0xBF)]),
        Utf8Sequence([rutf8(0xED, 0xED), rutf8(0x80, 0x9F), rutf8(0x80, 0xBF)]),
        Utf8Sequence([rutf8(0xEE, 0xEF), rutf8(0x80, 0xBF), rutf8(0x80, 0xBF)]),
    ]
