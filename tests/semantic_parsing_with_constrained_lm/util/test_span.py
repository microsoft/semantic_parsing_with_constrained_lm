# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=redefined-outer-name
import itertools
import random
from typing import List

import pytest

from semantic_parsing_with_constrained_lm.util.span import Span, SpanSet


class TestSpan:
    def test_malformed(self):
        with pytest.raises(AssertionError):
            Span(-1, 5)
        with pytest.raises(AssertionError):
            Span(4, 2)

    def test_len(self):
        assert len(Span(0, 4)) == 4

    def test_contains_pos(self):
        s1 = Span(0, 4)
        assert s1.contains_pos(0)
        assert s1.contains_pos(3)
        assert not s1.contains_pos(4)
        assert not s1.contains_pos(5)

    def test_contains(self):
        s1 = Span(0, 4)
        s2 = Span(1, 3)
        s3 = Span(2, 5)
        assert s1.contains(s1)
        assert s1.contains(s2)
        assert not s1.contains(s3)

    def test_crosses(self):
        s1 = Span(0, 4)
        s2 = Span(1, 3)
        s3 = Span(2, 5)
        s4 = Span(4, 6)
        assert not s1.crosses(s2)
        assert s1.crosses(s3)
        assert not s1.crosses(s4)


@pytest.fixture("module")
def all_span_sets_length_5() -> List[SpanSet]:
    result: List[SpanSet] = []
    for include in itertools.product((True, False), repeat=5):
        s = SpanSet()
        for i, b in zip(range(5), include):
            if b:
                s.add(Span(i, i + 1))
        result.append(s)
    return result


class TestSpanSet:
    def test_add_containing(self):
        s = SpanSet()
        s.add(Span(0, 4))
        s.add(Span(0, 3))
        assert s.spans == [Span(0, 4)]

        s = SpanSet()
        s.add(Span(1, 3))
        s.add(Span(0, 4))
        assert s.spans == [Span(0, 4)]

    def test_comprehensive(self):
        # Randomly try adding all spans between 0 and 5 in a shuffled order many times,
        # and check that .contains_pos behaves identically to a naive implementation.
        all_spans = [
            Span(left, right)
            for left in range(0, 5)
            for length in (1, 2, 3)
            for right in [left + length]
            if right <= 5
        ]
        rand = random.Random(1234)

        ordering = list(range(len(all_spans)))
        for _ in range(1000):
            rand.shuffle(ordering)
            added_spans = []
            s = SpanSet()
            for i in ordering:
                added_spans.append(all_spans[i])
                s.add(all_spans[i])
                for j in range(5):
                    assert s.contains_pos(j) == any(
                        span.contains_pos(j) for span in added_spans
                    )
            assert s.spans == [Span(0, 5)]

    def test_comprehensive_union(self, all_span_sets_length_5: List[SpanSet]):
        for s1, s2 in itertools.product(all_span_sets_length_5, repeat=2):
            s_union = s1.union(s2)
            assert set(s_union) == set(s1).union(set(s2))

    def test_comprehensive_difference(self, all_span_sets_length_5: List[SpanSet]):
        for s1, s2 in itertools.product(all_span_sets_length_5, repeat=2):
            s1_minus_s2 = s1.difference(s2)
            assert set(s1_minus_s2) == set(s1).difference(set(s2))
