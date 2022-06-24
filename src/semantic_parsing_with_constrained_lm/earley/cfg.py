# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ast
import collections
import functools
import itertools
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import blobfile
from blobfile import BlobFile
from lark import Lark, Transformer, v_args

from semantic_parsing_with_constrained_lm.earley.fsa_builders import (
    NFAFrag,
    re_alternative,
    re_atom,
    re_concat,
    re_kleene_star,
    re_optional,
    re_plus,
    re_ranges_span_set,
    re_repetitions,
    re_utf8,
)
from semantic_parsing_with_constrained_lm.earley.grammar import DFADottedRule, DFAGrammar, Nonterm
from semantic_parsing_with_constrained_lm.earley.unicode_categories_spans import category_to_span_set, raw_data
from semantic_parsing_with_constrained_lm.util.span import Span, SpanSet
from semantic_parsing_with_constrained_lm.util.util import identity


@functools.lru_cache(maxsize=None)
def parser() -> Lark:
    with open(Path(__file__).parent / "cfg.lark") as f:
        return Lark(f, start="start")


def parse_re_char_set(s: str) -> SpanSet:
    r"""Handles what appears inside [] in regular expressions.

    `s` should not contain the [] themselves, but only what's inside them.

    Handles the following constructs:
    - \p{...} (Unicode character category)
    - \uXXXX (Unicode character)
    - \UXXXXXXXX (Unicode character)
    - Regular characters
    - Escaped characters (\\, \-)
    - X-Y where X and Y are any of the above
    """

    offset = 0

    def peek() -> Optional[str]:
        if offset < len(s):
            return s[offset]
        else:
            return None

    def next_segment(length: int = 1, hint: str = "") -> str:
        nonlocal offset
        if offset + length > len(s):
            raise IndexError(
                f"Tried to read {length} characters at pos {offset}, but string ended early.\n"
                f"  {s}\n  {' ' * offset}^\n{hint}"
            )
        segment = s[offset : offset + length]
        offset += length
        return segment

    def read_elem_or_range() -> SpanSet:
        first = read_elem()
        if peek() == "-":
            read_elem()
            second = read_elem()
            assert not isinstance(first, SpanSet)
            assert not isinstance(second, SpanSet)
            assert first < second, f"{chr(first)} >= {chr(second)} in {s}"
            return SpanSet.new([Span.inclusive(first, second)])

        if isinstance(first, SpanSet):
            return first
        else:
            return SpanSet.new([Span.inclusive(first, first)])

    def read_elem() -> Union[int, SpanSet]:
        c = next_segment()
        if c == "\\" and peek() in ("-", "\\", "u", "U", "p"):
            return read_escaped()
        else:
            return ord(c)

    def read_escaped() -> Union[int, SpanSet]:
        c = next_segment()
        if c == "-":
            return ord("-")
        if c == "\\":
            return ord("\\")
        if c == "u":
            value = next_segment(4, "Expected 4 hexadecimal digits after \\u")
            try:
                return int(value, 16)
            except ValueError:
                # pylint: disable=raise-missing-from
                raise ValueError(f"Expected hexadecimal, but got {value!r}")
        if c == "U":
            value = next_segment(8, "Expected 8 hexadecimal digits after \\U")
            try:
                value_int = int(value, 16)
            except ValueError:
                # pylint: disable=raise-missing-from
                raise ValueError(f"Expected hexadecimal, but got {value!r}")
            if value_int > 0x10FFFF:
                raise ValueError(
                    f"Expected a codepoint in the range 0x0 to 0x10FFFF, but got {value_int:x}"
                )
            return value_int
        if c == "p":
            left_brace = next_segment()
            if left_brace != "{":
                raise ValueError("Expected '{' after 'p'")
            name = []
            while True:
                next_char = next_segment()
                if next_char == "}":
                    break
                name.append(next_char)

            name_str = "".join(name)
            try:
                return category_to_span_set(name_str)
            except KeyError as e:
                raise ValueError(
                    f"{name_str} is not a valid Unicode category."
                    f"Try one of: {', '.join(raw_data().keys())}"
                ) from e
        raise ValueError(c)

    all_span_sets = []
    while offset != len(s):
        all_span_sets.append(read_elem_or_range())

    if all_span_sets:
        return all_span_sets[0].union(*all_span_sets[1:])
    else:
        return SpanSet()


def load_grammar_from_directory(path: str, start_nt: str = "start") -> DFAGrammar:
    # TODO: Merge this blobfile.glob snippet with the one in read_grammar.py
    paths = set(
        itertools.chain(
            blobfile.glob(os.path.join(path, "**", "*.cfg")),
            blobfile.glob(os.path.join(path, "*.cfg")),
        )
    )
    if len(paths) == 0:
        raise FileNotFoundError(f"No .cfg files found in {path}")

    fragments: List[str] = []
    for grammar_path in paths:
        with BlobFile(grammar_path, streaming=False) as bf:
            fragments.append(bf.read())

    return load_grammar_from_fragments(fragments, start_nt)


def load_grammar_from_string(grammar: str, start_nt: str = "start") -> DFAGrammar:
    return load_grammar_from_fragments([grammar], start_nt)


def load_grammar_from_fragments(
    fragments: Iterable[str], start_nt: str = "start"
) -> DFAGrammar:
    uncompiled_rules: Dict[Nonterm, List[NFAFrag[Nonterm]]] = collections.defaultdict(
        list
    )
    for fragment in fragments:
        tree = parser().parse(fragment)
        for nonterm, expansions in cast(
            Sequence[Tuple[Nonterm, NFAFrag[Nonterm]]],
            CFGTransformer().transform(tree),
        ):
            uncompiled_rules[nonterm].append(expansions)

    compiled_rules: Dict[Nonterm, DFADottedRule] = {
        nonterm: DFADottedRule.from_rule(nonterm, re_alternative(*expansions))
        for nonterm, expansions in uncompiled_rules.items()
    }

    # TODO: Inline nonterminals used only once
    # TODO: Check that all nonterminals mentioned in the expansions actually exist
    return DFAGrammar(root=Nonterm(start_nt), expansions=compiled_rules)


@v_args(inline=True)
class CFGTransformer(Transformer):
    def start(
        self, *rules: Tuple[Nonterm, NFAFrag[Nonterm]]
    ) -> Sequence[Tuple[Nonterm, NFAFrag[Nonterm]]]:
        return rules

    def rule(
        self, nonterminal: Nonterm, expansion: NFAFrag[Nonterm]
    ) -> Tuple[Nonterm, NFAFrag[Nonterm]]:
        return (nonterminal, expansion)

    def expansion(self, *alts: NFAFrag[Nonterm]) -> NFAFrag[Nonterm]:
        return re_alternative(*alts)

    def alt(self, *elems: NFAFrag[Nonterm]) -> NFAFrag[Nonterm]:
        return re_concat(*elems)

    def empty(self) -> NFAFrag[Nonterm]:
        return identity

    def optional(self, elem: NFAFrag[Nonterm]) -> NFAFrag[Nonterm]:
        return re_optional(elem)

    def star(self, elem: NFAFrag[Nonterm]) -> NFAFrag[Nonterm]:
        return re_kleene_star(elem)

    def plus(self, elem: NFAFrag[Nonterm]) -> NFAFrag[Nonterm]:
        return re_plus(elem)

    def repeat_exact(self, elem: NFAFrag[Nonterm], count: int) -> NFAFrag[Nonterm]:
        return re_repetitions(elem, count, count)

    def repeat_min(self, elem: NFAFrag[Nonterm], min_: int) -> NFAFrag[Nonterm]:
        return re_repetitions(elem, min_, None)

    def repeat_max(self, elem: NFAFrag[Nonterm], max_: int) -> NFAFrag[Nonterm]:
        return re_repetitions(elem, 0, max_)

    def repeat_min_max(
        self, elem: NFAFrag[Nonterm], min_: int, max_: int
    ) -> NFAFrag[Nonterm]:
        return re_repetitions(elem, min_, max_)

    def char_class(self, s: str) -> NFAFrag[Nonterm]:
        return re_ranges_span_set(parse_re_char_set(s))

    def char_class_subtract(self, left: str, right: str) -> NFAFrag[Nonterm]:
        return re_ranges_span_set(
            parse_re_char_set(left).difference(parse_re_char_set(right))
        )

    def terminal(self, value: str) -> NFAFrag[Nonterm]:
        return re_utf8(ast.literal_eval(value))

    def nonterminal_lhs(self, name: str) -> Nonterm:
        return Nonterm(name)

    def nonterminal_rhs(self, name: str) -> NFAFrag[Nonterm]:
        return re_atom(Nonterm(name))

    count = int
