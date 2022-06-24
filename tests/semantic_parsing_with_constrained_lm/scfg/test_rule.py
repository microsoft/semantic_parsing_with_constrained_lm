# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
from lark import Lark

from semantic_parsing_with_constrained_lm.scfg.parser.parse import get_scfg_parser
from semantic_parsing_with_constrained_lm.scfg.parser.rule import MAYBE_PREFIX as PFX
from semantic_parsing_with_constrained_lm.scfg.parser.rule import (
    PlanRule,
    SyncRule,
    UtteranceRule,
    expand_optionals,
)
from semantic_parsing_with_constrained_lm.scfg.parser.token import (
    EmptyToken,
    NonterminalToken,
    TerminalToken,
)
from semantic_parsing_with_constrained_lm.scfg.read_grammar import parse_string


@pytest.fixture(scope="module", name="parser")
def create_parser() -> Lark:
    return get_scfg_parser("start_for_test")


def test_optional_symbols(parser: Lark):
    orig_rule = parse_string(
        parser, 'start -> yes? "the"? choice "one" , "blah" choice yes? no?'
    )
    expected = {
        SyncRule(
            lhs="start",
            utterance_rhss=(
                (
                    NonterminalToken(f"{PFX}_nt_yes", optional=False),
                    NonterminalToken(f"{PFX}_t_the", optional=False),
                    NonterminalToken("choice", optional=False),
                    TerminalToken('"one"', optional=False),
                ),
            ),
            plan_rhs=(
                TerminalToken('"blah"', optional=False),
                NonterminalToken("choice", optional=False),
                NonterminalToken(f"{PFX}_nt_yes", optional=False),
                NonterminalToken(f"{PFX}_nt_no", optional=False),
            ),
        ),
        # yes appears on both sides
        SyncRule(
            lhs=f"{PFX}_nt_yes",
            utterance_rhss=((NonterminalToken("yes", optional=False),),),
            plan_rhs=(NonterminalToken("yes", optional=False),),
        ),
        SyncRule(
            lhs=f"{PFX}_nt_yes",
            utterance_rhss=((EmptyToken(),),),
            plan_rhs=(EmptyToken(),),
        ),
        # "the" only appears on the utterance side
        UtteranceRule(
            lhs=f"{PFX}_t_the",
            utterance_rhss=((TerminalToken('"the"', optional=False),),),
        ),
        UtteranceRule(lhs=f"{PFX}_t_the", utterance_rhss=((EmptyToken(),),)),
        # no only appears on the plan side
        PlanRule(lhs=f"{PFX}_nt_no", rhs=(NonterminalToken("no", optional=False),)),
        PlanRule(lhs=f"{PFX}_nt_no", rhs=(EmptyToken(),)),
    }
    new_rules = expand_optionals(orig_rule)
    assert new_rules == expected
