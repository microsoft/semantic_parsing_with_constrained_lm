# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from abc import ABC
from dataclasses import dataclass, replace
from typing import Dict, Set, Tuple, cast

from semantic_parsing_with_constrained_lm.scfg.parser.token import (
    EmptyToken,
    NonterminalToken,
    OptionableSCFGToken,
    SCFGToken,
    TerminalToken,
)
from semantic_parsing_with_constrained_lm.scfg.parser.types import Expansion

MAYBE_PREFIX = "maybe__"


@dataclass(frozen=True)
class Rule(ABC):
    lhs: str


@dataclass(frozen=True)
class PlanRule(Rule):
    lhs: str
    rhs: Expansion


@dataclass(frozen=True)
class SyncRule(Rule):
    lhs: str
    utterance_rhss: Tuple[Expansion, ...]
    plan_rhs: Expansion


@dataclass(frozen=True)
class UtteranceRule(Rule):
    lhs: str
    utterance_rhss: Tuple[Expansion, ...]


# helpers
def term(s: str, optional=False) -> TerminalToken:
    return TerminalToken(underlying=json.dumps(s), optional=optional)


def nonterm(s: str, optional=False) -> NonterminalToken:
    return NonterminalToken(underlying=s, optional=optional)


def mirrored_rule(lhs: str, rhs: Expansion) -> SyncRule:
    """Creates a SyncRule with identical utterance and plan expansions."""
    return SyncRule(lhs=lhs, utterance_rhss=(rhs,), plan_rhs=rhs)


def expand_optionals(rule: Rule) -> Set[Rule]:
    """
    For each optional token `t` in rule, creates a fresh NT that expands to `t` or epsilon.
    """

    def clean(s: str) -> str:
        return "".join(c if c.isidentifier() else f"_chr{ord(c)}_" for c in s)

    def mk_optional_nt(s: OptionableSCFGToken) -> NonterminalToken:
        nt_or_t = "nt" if isinstance(s, NonterminalToken) else "t"
        return nonterm(f"{MAYBE_PREFIX}_{nt_or_t}_{clean(s.render())}")

    utterance_rhss = (
        rule.utterance_rhss if isinstance(rule, (SyncRule, UtteranceRule)) else ()
    )
    plan_rhss = (
        (rule.plan_rhs,)
        if isinstance(rule, SyncRule)
        else (rule.rhs,)
        if isinstance(rule, PlanRule)
        else ()
    )
    all_utterance_optionals: Set[OptionableSCFGToken] = {
        s
        for rhs in utterance_rhss
        for s in rhs
        if isinstance(s, OptionableSCFGToken) and s.optional
    }
    all_plan_optionals: Set[OptionableSCFGToken] = {
        s
        for rhs in plan_rhss
        for s in rhs
        if isinstance(s, OptionableSCFGToken) and s.optional
    }
    sync_optionals: Dict[SCFGToken, SCFGToken] = {
        s: mk_optional_nt(s)
        for s in all_utterance_optionals.intersection(all_plan_optionals)
    }
    utterance_only_optionals: Dict[SCFGToken, SCFGToken] = {
        s: mk_optional_nt(s)
        for s in all_utterance_optionals.difference(all_plan_optionals)
    }
    plan_only_optionals: Dict[SCFGToken, SCFGToken] = {
        s: mk_optional_nt(s)
        for s in all_plan_optionals.difference(all_utterance_optionals)
    }

    all_optionals: Dict[SCFGToken, SCFGToken] = {
        **utterance_only_optionals,
        **plan_only_optionals,
        **sync_optionals,
    }
    new_sync_rules: Set[Rule] = {
        r
        for s, nt in sync_optionals.items()
        for non_opt_s in [(replace(s, optional=False),)]
        for r in [
            mirrored_rule(nt.value, non_opt_s),
            mirrored_rule(nt.value, (EmptyToken(),)),
        ]
    }
    new_utt_rules: Set[Rule] = {
        r
        for s, nt in utterance_only_optionals.items()
        for non_opt_s in [(replace(s, optional=False),)]
        for r in [
            UtteranceRule(lhs=nt.value, utterance_rhss=(non_opt_s,)),
            UtteranceRule(lhs=nt.value, utterance_rhss=((EmptyToken(),),)),
        ]
    }
    new_plan_rules: Set[Rule] = {
        r
        for s, nt in plan_only_optionals.items()
        for non_opt_s in [(replace(s, optional=False),)]
        for r in [
            PlanRule(lhs=nt.value, rhs=non_opt_s),
            PlanRule(lhs=nt.value, rhs=(EmptyToken(),)),
        ]
    }
    new_maybe_rules: Set[Rule] = new_utt_rules | new_plan_rules | new_sync_rules

    def transform_rhs(rhs: Expansion) -> Expansion:
        return tuple(all_optionals.get(t, t) for t in rhs)

    new_main_rule: Rule
    if isinstance(rule, SyncRule):
        new_main_rule = SyncRule(
            lhs=rule.lhs,
            utterance_rhss=tuple(transform_rhs(rhs) for rhs in rule.utterance_rhss),
            plan_rhs=transform_rhs(rule.plan_rhs),
        )
    elif isinstance(rule, UtteranceRule):
        new_main_rule = UtteranceRule(
            lhs=rule.lhs,
            utterance_rhss=tuple(transform_rhs(rhs) for rhs in rule.utterance_rhss),
        )
    else:
        assert isinstance(rule, PlanRule), rule
        new_main_rule = PlanRule(lhs=rule.lhs, rhs=transform_rhs(rule.rhs))
    # pyright can't figure out that `new_main_rule` is a Rule
    return {cast(Rule, new_main_rule)} | new_maybe_rules
