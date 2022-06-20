# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from dataclasses import dataclass
from typing import List, Optional, Tuple

from dataflow.core.dialogue import Turn
from dataflow.core.lispress import (
    LET,
    META_CHAR,
    _is_named_arg,
    _named_arg_to_key,
    parse_lispress,
    try_round_trip,
)
from dataflow.core.sexp import Sexp, sexp_to_str

from semantic_parsing_with_constrained_lm.datum import FullDatum, FullDatumSub
from semantic_parsing_with_constrained_lm.eval import TopKExactMatch

STRING = "String"
REFERENCE = "Reference"
BOOLEAN = "Boolean"
NUMBER = "Number"
LONG = "Long"
TRUE = "true"
LAMBDA = "lambda"


@dataclass
class TopKLispressMatch(TopKExactMatch):
    def _is_correct(self, pred: str, target: FullDatumSub) -> bool:  # type: ignore
        """Can be overridden by child classes."""
        return try_round_trip(pred) == try_round_trip(target.canonical)


@dataclass(frozen=True)
class TurnV2(Turn):
    fully_typed_lispress: str = ""


@dataclass(frozen=True)
class DialogueV2:
    dialogue_id: str
    turns: List[TurnV2]

    def to_datums(self) -> List[FullDatum]:
        return [
            FullDatum(
                self.dialogue_id,
                t.turn_index,
                None,  # may want to include context in the future
                t.user_utterance.original_text,
                t.lispress,
            )
            for t in self.turns
        ]


@dataclass(frozen=True)
class TypeName:
    tpe: Sexp

    def __repr__(self) -> str:
        return sexp_to_str(self.tpe)


@dataclass(frozen=True)
class LispressExpr(abc.ABC):
    type: Optional[TypeName]


@dataclass(frozen=True)
class CallExpr(LispressExpr):
    instantiation_type: Optional[TypeName]
    name: str
    args: List[Tuple[str, LispressExpr]]


@dataclass(frozen=True)
class LetExpr(LispressExpr):
    var_assignments: List[Tuple[str, LispressExpr]]
    main_expr: LispressExpr


@dataclass(frozen=True)
class LambdaExpr(LispressExpr):
    var_type: TypeName
    var_name: str
    main_expr: LispressExpr


class LiteralExpr(LispressExpr, abc.ABC):
    pass


@dataclass(frozen=True)
class LongExpr(LiteralExpr):
    value: int


@dataclass(frozen=True)
class NumberExpr(LiteralExpr):
    value: float


@dataclass(frozen=True)
class StringExpr(LiteralExpr):
    value: str


@dataclass(frozen=True)
class BooleanExpr(LiteralExpr):
    value: bool


@dataclass(frozen=True)
class ReferenceExpr(LiteralExpr):
    var_name: str


def parse_fully_typed_lispress_v2(lispress: str) -> LispressExpr:
    sexp = parse_lispress(lispress)
    lispress_expr = parse_fully_typed_lispress_v2_sexp(sexp)
    return lispress_expr


def parse_fully_typed_lispress_v2_sexp(sexp: Sexp) -> LispressExpr:
    if isinstance(sexp, str):
        return ReferenceExpr(var_name=sexp, type=None)

    assert isinstance(sexp, list) and len(sexp) == 3, f"Failed to parse {sexp}"
    # Check for literals
    if sexp[1] == NUMBER and isinstance(sexp[2], str) and sexp[0] == META_CHAR:
        return NumberExpr(type=TypeName(tpe=NUMBER), value=float(sexp[2]))
    elif sexp[1] == LONG and isinstance(sexp[2], str) and sexp[0] == META_CHAR:
        return LongExpr(type=TypeName(tpe=LONG), value=int(sexp[2][:-1]))
    elif sexp[1] == STRING and isinstance(sexp[2], str) and sexp[0] == META_CHAR:
        return StringExpr(type=TypeName(tpe=STRING), value=sexp[2])
    elif sexp[1] == BOOLEAN and isinstance(sexp[2], str) and sexp[0] == META_CHAR:
        return BooleanExpr(type=TypeName(tpe=BOOLEAN), value=sexp[2] == TRUE)
    elif isinstance(sexp[2], str) and sexp[0] == META_CHAR:
        return ReferenceExpr(type=TypeName(tpe=REFERENCE), var_name=sexp[2])

    elif sexp[0] == LAMBDA:
        assert len(sexp[1]) == 1 and len(sexp[1][0]) == 3 and sexp[1][0][0] == META_CHAR
        var_type = TypeName(tpe=sexp[1][0][1])
        var_name = sexp[1][0][2]
        main_expr = sexp[2]
        return LambdaExpr(
            var_name=var_name,  # type: ignore
            var_type=var_type,
            main_expr=parse_fully_typed_lispress_v2_sexp(main_expr),
            type=None,
        )

    # Check for let expression
    elif sexp[0] == LET and isinstance(sexp[1], list) and isinstance(sexp[2], list):
        var_assignment_list = sexp[1]
        assert len(var_assignment_list) % 2 == 0
        assert all([isinstance(item, str) for item in var_assignment_list[0::2]])
        var_assignments = [
            (
                var_assignment_list[index],
                parse_fully_typed_lispress_v2_sexp(var_assignment_list[index + 1]),
            )
            for index in range(0, len(var_assignment_list), 2)
        ]
        main_expr = sexp[2]
        return LetExpr(
            var_assignments=var_assignments,  # type: ignore
            main_expr=parse_fully_typed_lispress_v2_sexp(main_expr),
            type=None,
        )

    else:
        # Check for CallExp
        if isinstance(sexp[2], list) and len(sexp[2]) > 0 and sexp[0] == META_CHAR:
            return_type = TypeName(tpe=sexp[1])
            function_name = None
            function_instantiation_type = None
            function_sexp, *key_value_args = sexp[2]
            if isinstance(function_sexp, str):
                function_name = function_sexp
            elif (
                isinstance(function_sexp, list)
                and len(function_sexp) == 3
                and function_sexp[0] == META_CHAR
                and isinstance(function_sexp[2], str)
            ):
                function_instantiation_type = TypeName(tpe=function_sexp[1])
                function_name = function_sexp[2]

            assert function_name is not None
            assert len(key_value_args) % 2 == 0 and all(
                [
                    isinstance(item, str) and _is_named_arg(item)
                    for item in key_value_args[0::2]
                ]
            )

            args = []
            for index in range(0, len(key_value_args), 2):
                key = _named_arg_to_key(key_value_args[index])  # type: ignore
                value = parse_fully_typed_lispress_v2_sexp(key_value_args[index + 1])
                args.append((key, value))

            return CallExpr(
                type=return_type,
                name=function_name,
                args=args,
                instantiation_type=function_instantiation_type,
            )

    raise ValueError(f"Could not parse: {sexp}")
