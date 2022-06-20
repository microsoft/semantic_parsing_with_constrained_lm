# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import functools
import re
from typing import Iterable, List, Optional, Set

from semantic_parsing_with_constrained_lm.domains import dfa_grammar_utils
from semantic_parsing_with_constrained_lm.domains.lispress_v2.lispress_exp import (
    BooleanExpr,
    CallExpr,
    DialogueV2,
    LambdaExpr,
    LetExpr,
    LispressExpr,
    LongExpr,
    NumberExpr,
    ReferenceExpr,
    StringExpr,
    TypeName,
    parse_fully_typed_lispress_v2,
)


def get_nt_from_type(type_name: TypeName) -> str:
    segments = (
        str(type_name)
        .replace(" ", " SP ")
        .replace("(", " LP ")
        .replace(")", " RP ")
        .replace(".", " DOT ")
        .split()
    )
    return "_".join(segments + ["NT"])


def extract_grammar_rules(lispress_expr: LispressExpr) -> Set[str]:
    lhs = get_nt_from_type(lispress_expr.type)  # type: ignore
    rules = set()

    if isinstance(lispress_expr, (NumberExpr, LongExpr, StringExpr, BooleanExpr)):
        pass
    elif isinstance(lispress_expr, ReferenceExpr):
        rules.add(f'{lhs} -> "{lispress_expr.var_name}"')

    elif isinstance(lispress_expr, LambdaExpr):
        rhs_items = [
            f'"(lambda (^{str(lispress_expr.var_type)} {lispress_expr.var_name}) "',
            get_nt_from_type(lispress_expr.main_expr.type),  # type: ignore
            '")"',
        ]
        rhs = " ".join(rhs_items)
        rules.add(f"{lhs} -> {rhs}")
        rules.update(extract_grammar_rules(lispress_expr.main_expr))

    elif isinstance(lispress_expr, LetExpr):
        var_name_expr_nts = []
        for var_name, var_expr in lispress_expr.var_assignments:
            var_name_expr_nts.extend([f'"{var_name}"', get_nt_from_type(var_expr.type)])  # type: ignore
            rules.update(extract_grammar_rules(var_expr))
        var_name_expr_nts_str = ' " " '.join(var_name_expr_nts)
        rhs = f'"(let (" {var_name_expr_nts_str} ") " {get_nt_from_type(lispress_expr.main_expr.type)} ")"'  # type: ignore
        rules.add(f"{lhs} -> {rhs}")
        rules.update(extract_grammar_rules(lispress_expr.main_expr))

    elif isinstance(lispress_expr, CallExpr):
        rhs_items: List[str] = []
        if lispress_expr.instantiation_type is not None:
            rhs_items.append(f'"^{lispress_expr.instantiation_type} "?')

        rhs_items.append(f'"{lispress_expr.name}"')

        for k, v in lispress_expr.args:
            rhs_items.extend([f'" :{k}"?', '" "', get_nt_from_type(v.type)])  # type: ignore
            rules.update(extract_grammar_rules(v))

        rhs = " ".join(rhs_items)
        rules.add(f'{lhs} -> "(" {rhs} ")"')

    return rules


def extract_grammar(
    dataflow_dialogues: Iterable[DialogueV2],
    whitelisted_dialogue_ids: Optional[Set[str]] = None,
) -> Set[str]:
    grammar_rules = set()
    for dataflow_dialogue in dataflow_dialogues:
        if (
            whitelisted_dialogue_ids is not None
            and dataflow_dialogue.dialogue_id not in whitelisted_dialogue_ids
        ):
            continue

        for turn in dataflow_dialogue.turns:
            lispress_expr = parse_fully_typed_lispress_v2(turn.fully_typed_lispress)
            grammar_rules.update(extract_grammar_rules(lispress_expr))
            root_type_nt = get_nt_from_type(lispress_expr.type)  # type: ignore
            grammar_rules.add(f'start -> " " {root_type_nt}')

            # find string literals
            for match in re.finditer(r'Path\.apply "([^"]*)"', turn.lispress):
                start = match.start(1)
                end = match.end(1)
                item = turn.lispress[start:end]
                # We use `repr` because the .cfg parser uses `ast.literal_eval`
                # to parse the strings, since that will handle backslash escape
                # sequences. Without `repr` the resulting grammar will have one
                # level of escaping removed.
                grammar_rules.add(f"path_literal -> {repr(item)}")
    grammar_rules.update(
        [
            'Boolean_NT -> "true"',
            'Boolean_NT -> "false"',
            r'String_NT -> "\"" (String_NT_content | path_literal | "output" | "place" | "start") "\""',
            # Lispress V2 string literals are JSON string literals, so we follow this grammar:
            # https://datatracker.ietf.org/doc/html/rfc8259#section-7
            r'String_NT_content -> ([[\u0020-\U0010FFFF]--[\u0022\u005C]] | "\\" (["\u005C/bfnrt] | u[0-9A-Fa-f]{4}))*',
            'Number_NT -> ("0" | [1-9][0-9]*) ("." [0-9]+)?',
            'Long_NT -> ("0" | [1-9][0-9]*) "L"',
        ]
    )
    return grammar_rules


create_partial_parse_builder = functools.partial(
    dfa_grammar_utils.create_partial_parse_builder,
    utterance_nonterm_name="String_NT_content",
)
