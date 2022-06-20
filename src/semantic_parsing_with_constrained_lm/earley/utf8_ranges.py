# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The code in this file is translated directly from
# https://github.com/rust-lang/regex/blob/258bdf798a14f50529c1665e84cc8a3a9e2c90fc/regex-syntax/src/utf8.rs
# The docstring here, at the top of the file, is copied directly from there.
# Most other docstrings are also copied directly from the comments for the corresponding functions.
"""
Converts ranges of Unicode scalar values to equivalent ranges of UTF-8 bytes.

This is sub-module is useful for constructing byte based automatons that need
to embed UTF-8 decoding. The most common use of this module is in conjunction
with the [`hir::ClassUnicodeRange`](../hir/struct.ClassUnicodeRange.html) type.

See the documentation on the `Utf8Sequences` iterator for more details and
an example.

# Wait, what is this?

This is simplest to explain with an example. Let's say you wanted to test
whether a particular byte sequence was a Cyrillic character. One possible
scalar value range is `[0400-04FF]`. The set of allowed bytes for this
range can be expressed as a sequence of byte ranges:

```text
[D0-D3][80-BF]
```

This is simple enough: simply encode the boundaries, `0400` encodes to
`D0 80` and `04FF` encodes to `D3 BF`, and create ranges from each
corresponding pair of bytes: `D0` to `D3` and `80` to `BF`.

However, what if you wanted to add the Cyrillic Supplementary characters to
your range? Your range might then become `[0400-052F]`. The same procedure
as above doesn't quite work because `052F` encodes to `D4 AF`. The byte ranges
you'd get from the previous transformation would be `[D0-D4][80-AF]`. However,
this isn't quite correct because this range doesn't capture many characters,
for example, `04FF` (because its last byte, `BF` isn't in the range `80-AF`).

Instead, you need multiple sequences of byte ranges:

```text
[D0-D3][80-BF]  # matches codepoints 0400-04FF
[D4][80-AF]     # matches codepoints 0500-052F
```

This gets even more complicated if you want bigger ranges, particularly if
they naively contain surrogate codepoints. For example, the sequence of byte
ranges for the basic multilingual plane (`[0000-FFFF]`) look like this:

```text
[0-7F]
[C2-DF][80-BF]
[E0][A0-BF][80-BF]
[E1-EC][80-BF][80-BF]
[ED][80-9F][80-BF]
[EE-EF][80-BF][80-BF]
```

Note that the byte ranges above will *not* match any erroneous encoding of
UTF-8, including encodings of surrogate codepoints.

And, of course, for all of Unicode (`[000000-10FFFF]`):

```text
[0-7F]
[C2-DF][80-BF]
[E0][A0-BF][80-BF]
[E1-EC][80-BF][80-BF]
[ED][80-9F][80-BF]
[EE-EF][80-BF][80-BF]
[F0][90-BF][80-BF][80-BF]
[F1-F3][80-BF][80-BF][80-BF]
[F4][80-8F][80-BF][80-BF]
```

This module automates the process of creating these byte ranges from ranges of
Unicode scalar values.

# Lineage

I got the idea and general implementation strategy from Russ Cox in his
[article on regexps](https://web.archive.org/web/20160404141123/https://swtch.com/~rsc/regexp/regexp3.html) and RE2.
Russ Cox got it from Ken Thompson's `grep` (no source, folk lore?).
I also got the idea from
[Lucene](https://github.com/apache/lucene-solr/blob/ae93f4e7ac6a3908046391de35d4f50a0d3c59ca/lucene/core/src/java/org/apache/lucene/util/automaton/UTF32ToUTF8.java),
which uses it for executing automata on their term index.
"""


from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from semantic_parsing_with_constrained_lm.util.span import Span

MAX_UTF8_BYTES = 4


@dataclass
class Utf8Sequence:
    """Utf8Sequence represents a sequence of byte ranges.

    To match a Utf8Sequence, a candidate byte sequence must match each
    successive range.

    For example, if there are two ranges, `[C2-DF][80-BF]`, then the byte
    sequence `\\xDD\\x61` would not match because `0x61 < 0x80`.
    """

    ranges: "List[Utf8Range]"

    def __post_init__(self):
        assert 1 <= len(self.ranges) <= 4

    @staticmethod
    def from_encoded_range(start: Sequence[np.uint8], end: Sequence[np.uint8]):
        """Creates a new UTF-8 sequence from the encoded bytes of a scalar value
        range.

        This assumes that `start` and `end` have the same length.
        """
        assert len(start) == len(end)
        assert 2 <= len(start) <= 4
        return Utf8Sequence([Utf8Range(s, e) for s, e in zip(start, end)])

    # Unimplemented: as_slice

    def __len__(self) -> int:
        """Returns the number of byte ranges in this sequence.

        The length is guaranteed to be in the closed interval `[1, 4]`."""

        return len(self.ranges)

    def reverse(self) -> "Utf8Sequence":
        """Reverses the ranges in this sequence.

        For example, if this corresponds to the following sequence:

        ```text
        [D0-D3][80-BF]
        ```

        Then after reversal, it will be

        ```text
        [80-BF][D0-D3]
        ```

        This is useful when one is constructing a UTF-8 automaton to match
        character classes in reverse."""
        return Utf8Sequence(list(reversed(self.ranges)))

    def matches(self, byte_seq: Sequence[np.uint8]) -> bool:
        """Returns true if and only if a prefix of `bytes` matches this sequence
        of byte ranges.
        """
        if len(byte_seq) < len(self):
            return False

        for b, r in zip(byte_seq, self.ranges):
            if not r.matches(b):
                return False

        return True

    def __repr__(self) -> str:
        return "".join(repr(r) for r in self.ranges)


@dataclass
class Utf8Range:
    """A single inclusive range of UTF-8 bytes."""

    # Start of byte range (inclusive).
    start: np.uint8

    # End of byte range (inclusive).
    end: np.uint8

    def matches(self, b: np.uint8) -> bool:
        """Returns true if and only if the given byte is in the range."""
        return bool(self.start <= b <= self.end)

    def __repr__(self) -> str:
        if self.start == self.end:
            return f"[{self.start:02X}]"
        else:
            return f"[{self.start:02X}-{self.end:02X}]"


@dataclass
class ScalarRange:
    """A single inclusive range of two integers (meant to be uint32).

    This is used to represent a range of Unicode code points."""

    start: int
    end: int

    def split(self) -> "Optional[Tuple[ScalarRange, ScalarRange]]":
        """split splits this range if it overlaps with a surrogate codepoint.

        Either or both ranges may be invalid."""

        # 0xE000 to 0xD7FF are surrogate code points
        if self.start < 0xE000 and self.end > 0xD7FF:
            return (ScalarRange(self.start, 0xD7FF), ScalarRange(0xE000, self.end))
        else:
            return None

    def is_valid(self) -> bool:
        """is_valid returns true if and only if start <= end."""
        return self.start <= self.end

    def as_ascii(self) -> Optional[Utf8Range]:
        """as_ascii returns this range as a Utf8Range if and only if all scalar
        values in this range can be encoded as a single byte."""

        if self.is_ascii():
            return Utf8Range(np.uint8(self.start), np.uint8(self.end))
        else:
            return None

    def is_ascii(self) -> bool:
        """is_ascii returns true if the range is ASCII only (i.e., takes a single
        byte to encode any scalar value)."""

        return self.is_valid() and self.end <= 0x7F

    def encode(self) -> Tuple[Sequence[np.uint8], Sequence[np.uint8]]:
        """encode writes the UTF-8 encoding of the start and end of this range
        and returns them as a pair.

        The two return values will have maximum length `MAX_UTF8_BYTES`."""

        return (
            np.frombuffer(chr(self.start).encode("utf-8"), dtype=np.uint8),
            np.frombuffer(chr(self.end).encode("utf-8"), dtype=np.uint8),
        )


@dataclass
class Utf8Sequences:
    """An iterator over ranges of matching UTF-8 byte sequences.

    The iteration represents an alternation of comprehensive byte sequences
    that match precisely the set of UTF-8 encoded scalar values.

    A byte sequence corresponds to one of the scalar values in the range given
    if and only if it completely matches exactly one of the sequences of byte
    ranges produced by this iterator.

    Each sequence of byte ranges matches a unique set of bytes. That is, no two
    sequences will match the same bytes.

    # Example

    This shows how to match an arbitrary byte sequence against a range of
    scalar values.

    ```python
    def matches(seqs: List[Utf8Sequence], bytes_: bytes) -> bool {
        arr = np.frombuffer(bytes_, dtype=np.uint8)
        for range in seqs:
            if range.matches(arr):
                return True
        return False
    }

    # Test the basic multilingual plane.
    seqs = list(Utf8Sequences.new('\\u0000', '\\uFFFF'))

    # UTF-8 encoding of 'a'.
    assert matches(seqs, b"\\x61"))
    # UTF-8 encoding of '☃' (`\\u{2603}`).
    assert matches(seqs, b"\\xE2\x98\x83"))
    # UTF-8 encoding of `\\u{10348}` (outside the BMP).
    assert not matches(seqs, b"\\xF0\\x90\\x8D\\x88"))
    # Tries to match against a UTF-8 encoding of a surrogate codepoint,
    # which is invalid UTF-8, and therefore fails, despite the fact that
    # the corresponding codepoint (0xD800) falls in the range given.
    assert not matches(seqs, b"\\xED\\xA0\\x80"))
    # And fails against plain old invalid UTF-8.
    assert not matches(seqs, b"\\xFF\\xFF")
    ```

    If this example seems circuitous, that's because it is! It's meant to be
    illustrative. In practice, you could just try to decode your byte sequence
    and compare it with the scalar value range directly. However, this is not
    always possible (for example, in a byte based automaton)."""

    range_stack: List[ScalarRange]

    @staticmethod
    def new(start: str, end: str) -> "Utf8Sequences":
        """Create a new iterator over UTF-8 byte ranges for the scalar value
        range given.
        """
        assert len(start) == 1 and len(end) == 1
        return Utf8Sequences([ScalarRange(ord(start), ord(end))])

    @staticmethod
    def new_int(start: int, end: int) -> "Utf8Sequences":
        """start and end are inclusive"""
        return Utf8Sequences([ScalarRange(start, end)])

    @staticmethod
    def from_span(span: Span) -> "Utf8Sequences":
        # span.end - 1 because ScalarRange uses closed intervals, i.e., [begin, end]
        # but Span uses half-open intervals, i.e., [begin, end)
        return Utf8Sequences([ScalarRange(span.begin, span.end - 1)])

    def push(self, start: int, end: int) -> None:
        self.range_stack.append(ScalarRange(start, end))

    def __iter__(self) -> Iterator[Utf8Sequence]:
        return self

    def __next__(self) -> Utf8Sequence:
        while len(self.range_stack):  # 'TOP
            r = self.range_stack.pop()
            while True:  # 'INNER
                maybe_split = r.split()
                if maybe_split is not None:
                    r1, r2 = maybe_split
                    self.push(r2.start, r2.end)
                    r.start = r1.start
                    r.end = r1.end
                    continue

                if not r.is_valid():
                    break  # originally, continue 'TOP

                continue_after = False
                for i in range(1, MAX_UTF8_BYTES):
                    max_for_i = max_scalar_value(i)
                    if r.start <= max_for_i < r.end:
                        self.push(max_for_i + 1, r.end)
                        r.end = max_for_i
                        continue_after = True
                        break
                if continue_after:
                    continue

                maybe_ascii_range = r.as_ascii()
                if maybe_ascii_range is not None:
                    return Utf8Sequence([maybe_ascii_range])

                continue_after = False
                for i in range(1, MAX_UTF8_BYTES):
                    m = (1 << (6 * i)) - 1
                    if (r.start & ~m) != (r.end & ~m):
                        if (r.start & m) != 0:
                            self.push((r.start | m) + 1, r.end)
                            r.end = r.start | m
                            continue_after = True
                            break
                        if (r.end & m) != m:
                            self.push(r.end & ~m, r.end)
                            r.end = (r.end & ~m) - 1
                            continue_after = True
                            break
                if continue_after:
                    continue

                start, end = r.encode()
                return Utf8Sequence.from_encoded_range(start, end)

        raise StopIteration


def max_scalar_value(nbytes: int) -> int:
    return {1: 0x007F, 2: 0x07FF, 3: 0xFFFF, 4: 0x10FF}[nbytes]
