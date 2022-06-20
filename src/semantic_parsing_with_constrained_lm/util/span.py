# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
import operator
from typing import AbstractSet, Any, Iterable, Iterator, List, Optional, Tuple

from semantic_parsing_with_constrained_lm.util.util import bisect_right


class Span(AbstractSet[int]):
    """A span consisting of two integers. `begin` is inclusive and `end` is exclusive.

    See also [[epic.trees.Span]].
    """

    __slots__ = ("begin", "end")

    @staticmethod
    def inclusive(begin: int, end: int) -> "Span":
        return Span(begin, end + 1)

    def __init__(self, begin: int, end: int):
        assert 0 <= begin <= end
        self.begin = begin
        self.end = end

    def __len__(self) -> int:
        return self.end - self.begin

    def contains_pos(self, pos: int) -> bool:
        return self.begin <= pos < self.end

    def contains(self, other: "Span") -> bool:
        """Returns true if `other` is entirely contained within `self`."""
        return self.begin <= other.begin and self.end >= other.end

    def crosses(self, other: "Span") -> bool:
        """Returns true if `self` and `other` overlap but containment or equality does not hold."""
        return (
            self.begin < other.begin and self.end < other.end and self.end > other.begin
        ) or (
            other.begin < self.begin and other.end < self.end and other.end > self.begin
        )

    def overlaps(self, other: "Span") -> bool:
        """Returns true if any overlap exists between `self` and `other`."""
        return self.contains(other) or other.contains(self) or self.crosses(other)

    def adjacent_to(self, other: "Span") -> bool:
        return self.begin == other.end or self.end == other.begin

    def astuple(self) -> Tuple[int, int]:
        return self.begin, self.end

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.begin, self.end))

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Span)
            and self.begin == other.begin
            and self.end == other.end
        )

    def __hash__(self) -> int:
        return hash(self.astuple())

    def __repr__(self) -> str:
        return f"Span({self.begin}, {self.end})"

    def __contains__(self, x: Any) -> bool:
        return isinstance(x, int) and self.contains_pos(x)


class SpanSet(AbstractSet[int]):
    """Keeps track of a set of spans. Insert-only.

    Conceptually, this class maintains a bitset of unbounded length,
    where all bits are initially set to 0.
    When a span is added, the bits at indices within the span are set to 1.

    For efficiency, it doesn't actually maintain such a bitset,
    but rather a sorted list of coalesced spans.
    """

    __slots__ = ("spans",)

    def __init__(self, spans: Iterable[Span] = ()):
        # Kept sorted by .begin. Contains no overlaps.
        self.spans: List[Span] = list(spans)
        if self.spans:
            assert all(
                first.end < second.begin
                for first, second in zip(self.spans, self.spans[1:])
            ), f"{self.spans} does not meet invariant"

    @staticmethod
    def new(spans: Iterable[Span]):
        result = SpanSet()
        for span in spans:
            result.add(span)
        return result

    def add(self, span: Span) -> None:
        # Figure out which existing spans need to be coalesced with this span,
        # and where to insert it, in order to maintain the invariant.
        # We need to find all existing spans that either have overlap with `span`,
        # or are adjacent to `span`.
        # We will remove all of these, create a big span that exactly covers the same range
        # as them and the input `span`, and insert it into the list.

        # Empty spans don't affect the set.
        if len(span) == 0:
            return

        # 1. Find the place where we should start looking in `self.spans`.
        insert_index = bisect_right(self.spans, span, key=operator.attrgetter("begin"))

        # 2. Determine the indices of the existing spans which overlap with `span` or are adjacent.
        # Because the spans are in sorted order, we just need a min and max (inclusive),
        # and everything in between will satisfy the property.
        def get_last(indices: Iterable[int]) -> Optional[int]:
            result = None
            for i in indices:
                other = self.spans[i]
                if span.overlaps(other) or span.adjacent_to(other):
                    result = i
                else:
                    break
            return result

        min_adj_or_overlap_index = get_last(range(insert_index - 1, -1, -1))
        max_adj_or_overlap_index = get_last(range(insert_index, len(self.spans)))

        # 3. Create the new span, coalescing existing ones, and insert it in the list.
        if min_adj_or_overlap_index is None and max_adj_or_overlap_index is None:
            # Nothing overlapped, so we can just add this span without changing any existing ones..
            self.spans.insert(insert_index, span)
        else:
            # Find the spans that we need to coalesce.
            # We can't directly use `min_adj_or_overlap_index` and `max_adj_or_overlap_index`
            # because they may be None.

            # `left_bound` is the minimum index in self.spans, of the spans that we need to coalesce with `span`.
            # We can prove that 0 <= left_bound < len(self.spans).
            #   First, we have that 0 <= insert_index <= len(self.spans) since it comes from `bisect_right`.
            #   Case 1. min_adj_overlap_index == None.
            #     Since left_bound := insert_index, We need to show that insert_index != len(self.spans).
            #     Assume insert_index == len(self.spans).
            #     Then span.begin >= self.spans[-1].begin, i.e. nothing is to the right.
            #     Therefore max_adj_or_overlap_index must be None.
            #     But min_adj_overlap_index == None, so we have a contradiction
            #     (both can't be None since we are in the else clause of the if statement).
            #   Case 2. min_adj_overlap_index != None.
            #     By construction, 0 <= min_adj_overlap_index < insert_index <= len(self.spans).
            #     left_bound := min_overlap_index, so we are done.
            left_bound = (
                insert_index
                if min_adj_or_overlap_index is None
                else min_adj_or_overlap_index
            )

            # Maximum index in self.spans, of the spans that we need to coalesce with `span`.
            right_bound = (
                insert_index - 1
                if max_adj_or_overlap_index is None
                else max_adj_or_overlap_index
            )

            # Create the new coalesced span, and add it to the list.
            new_span = Span(
                min(self.spans[left_bound].begin, span.begin),
                max(self.spans[right_bound].end, span.end),
            )
            self.spans[left_bound] = new_span
            del self.spans[left_bound + 1 : right_bound + 1]

    def contains_pos(self, pos: int) -> bool:
        return any(span.contains_pos(pos) for span in self.spans)

    def contains(self, other: "Span") -> bool:
        return any(span.contains(other) for span in self.spans)

    def overlaps(self, other: "Span") -> bool:
        return any(span.overlaps(other) for span in self.spans)

    def __bool__(self) -> bool:
        return bool(self.spans)

    def __len__(self) -> int:
        return sum(len(span) for span in self.spans)

    def __iter__(self) -> Iterator[int]:
        return itertools.chain.from_iterable(iter(span) for span in self.spans)

    def __contains__(self, x: Any) -> bool:
        return isinstance(x, int) and self.contains_pos(x)

    def union(self, *others: "SpanSet") -> "SpanSet":
        return SpanSet.new(
            itertools.chain(self.spans, *(other.spans for other in others))
        )

    def difference(self, *others: "SpanSet") -> "SpanSet":
        span_list = self.spans
        new_span_list = []

        for other in others:
            my_span_iter = iter(span_list)
            other_span_iter = iter(other.spans)

            my_span = next(my_span_iter, None)
            other_span = next(other_span_iter, None)

            while True:
                if my_span is None:
                    break
                if other_span is None:
                    new_span_list.append(my_span)
                    new_span_list.extend(my_span_iter)
                    break

                # See how the two spans are related
                # Case 1
                # [   ]         <-- my_span
                #        [    ] <-- other_span
                if my_span.end <= other_span.begin:
                    new_span_list.append(my_span)
                    my_span = next(my_span_iter, None)
                    continue
                # Case 2: inverse of case 1
                #        [    ] <-- my_span
                # [   ]         <-- other_span
                if other_span.end <= my_span.begin:
                    other_span = next(other_span_iter, None)
                    continue

                if my_span.begin < other_span.begin:
                    # Case 3
                    # [   ]    <-- my_span
                    #   [    ] <-- other_span
                    if my_span.end <= other_span.end:
                        new_span_list.append(Span(my_span.begin, other_span.begin))
                        my_span = next(my_span_iter, None)
                    # Case 4
                    # [        ] <-- my_span
                    #   [   ]    <-- other_span
                    else:
                        new_span_list.append(Span(my_span.begin, other_span.begin))

                        my_span = Span(other_span.end, my_span.end)
                        other_span = next(other_span_iter, None)
                    continue

                # Case 5: inverse of case 4
                #   [   ]    <-- my_span
                # [        ] <-- other_span
                # ====
                # [   ]      <-- my_span
                # [        ] <-- other_span
                if my_span.end <= other_span.end:
                    my_span = next(my_span_iter, None)
                # Case 6: inverse of case 3
                #   [    ] <-- my_span
                # [   ]    <-- other_span
                # ===
                # [      ] <-- my_span
                # [   ]    <-- other_span
                else:
                    my_span = Span(other_span.end, my_span.end)
                    other_span = next(other_span_iter, None)

            span_list = new_span_list

        return SpanSet(span_list)

    def __repr__(self) -> str:
        return f"SpanSet({self.spans})"

    __ior__ = add
