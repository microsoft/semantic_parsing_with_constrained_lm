# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Provides data about which Unicode characters belong to which general category.

The concept is explained here:
- https://www.unicode.org/reports/tr44/#General_Category_Values
- https://en.wikipedia.org/wiki/Unicode_character_property#General_Category

unicode_categories.json was created by translating
https://raw.githubusercontent.com/rust-lang/regex/258bdf798a14f50529c1665e84cc8a3a9e2c90fc/regex-syntax/src/unicode_tables/general_category.rs
"""
import functools
import json
from pathlib import Path
from typing import Dict, List

from semantic_parsing_with_constrained_lm.util.span import Span, SpanSet


@functools.lru_cache(maxsize=None)
def raw_data() -> Dict[str, List[List[int]]]:
    with open(Path(__file__).absolute().parent / "unicode_categories.json") as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def category_to_span_set(name: str) -> SpanSet:
    """Returns the SpanSet for the category name.

    The SpanSet contains the Unicode code points for the corresponding general category.
    Only long names with underscores (e.g. "Letter", "Cased_Letter") are accepted."""

    return SpanSet(Span(x, y) for x, y in raw_data()[name])
