# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import contextlib
import itertools
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple, Union

import blobfile
from blobfile import BlobFile
from cached_property import cached_property

from semantic_parsing_with_constrained_lm.util.types import StrPath
from semantic_parsing_with_constrained_lm.scfg.parser.macro import Macro, expand_macros
from semantic_parsing_with_constrained_lm.scfg.parser.parse import get_scfg_parser, parse_string
from semantic_parsing_with_constrained_lm.scfg.parser.rule import (
    Rule,
    SyncRule,
    UtteranceRule,
    expand_optionals,
)
from semantic_parsing_with_constrained_lm.scfg.parser.types import Expansion, Nonterminal
from semantic_parsing_with_constrained_lm.scfg.parser.utils import is_skippable

GrammarRules = Dict[Nonterminal, Set[Expansion]]


def find_all_scfg_paths(folder_path: StrPath) -> List[str]:
    """Finds all file paths under the given path with an .scfg extension."""
    paths = set()
    paths.update(
        # Both patterns needed to work with azure blob storage
        list(blobfile.glob(os.path.join(folder_path, "**", "*.scfg")))
        + list(blobfile.glob(os.path.join(folder_path, "*.scfg")))
    )
    return list(paths)


@dataclass
class PreprocessedGrammar:
    sync_rules: Dict[Tuple[Nonterminal, Expansion], Set[Expansion]]
    utterance_rules: GrammarRules

    @cached_property
    def all_utterance_rules(self) -> GrammarRules:
        """Merges `utterance_rules` with the utterance part of `sync_rules`."""
        result: DefaultDict[Nonterminal, Set[Expansion]] = defaultdict(set)
        for (lhs, _plan_rhs), rhss in self.sync_rules.items():
            result[lhs] |= rhss
        for lhs, rhss in self.utterance_rules.items():
            result[lhs] |= rhss
        return dict(result)

    @cached_property
    def all_plan_rules(self) -> GrammarRules:
        """
        Extracts the plan part of `sync_rules`
        (there is no stand-alone `plan_rules`).
        """
        result: Dict[Nonterminal, Set[Expansion]] = defaultdict(set)
        for (lhs, plan_rhs), _ in self.sync_rules.items():
            result[lhs].add(plan_rhs)
        return dict(result)

    def merge(self, other: "PreprocessedGrammar") -> "PreprocessedGrammar":
        """
        Non-destructively creates a new `PreprocessedGrammar` whose rules
        and macros are the union of `self`'s and `other`'s.
        """
        sync_rules = {k: set(v) for k, v in self.sync_rules.items()}
        utterance_rules = {k: set(v) for k, v in self.utterance_rules.items()}
        for nt_and_utt_rhs, plan_rhss in other.sync_rules.items():
            sync_rules.setdefault(nt_and_utt_rhs, set()).update(plan_rhss)
        for nt, plan_rhss in other.utterance_rules.items():
            utterance_rules.setdefault(nt, set()).update(plan_rhss)
        return PreprocessedGrammar(
            sync_rules=sync_rules, utterance_rules=utterance_rules
        )

    @classmethod
    def from_line_iter(cls, grammar_input: Iterable[str]) -> "PreprocessedGrammar":
        """
        Reads an .scfg stream with lines that define:

        Sync rules: nonterminal -> utterance1 | ... | utteranceN , plan
        OR
        Utterance rules: nonterminal 1> utterance1 | ... | utteranceN
        OR
        Plan macro rules: nonterminal 2> plan
                          nonterminal(...) 2> plan

        and stores them in dictionaries.

        For sync rules, we key on (nonterminal, plan) so that we can associate all the utterances that have the same plan
        with each other.
        """
        parser = get_scfg_parser()
        macros: Dict[str, Macro] = {}
        rules: List[Rule] = []
        for line in grammar_input:
            line = line.strip()
            if is_skippable(line):
                continue
            rule: Union[Rule, Macro] = parse_string(parser, line)
            if isinstance(rule, Macro):
                assert (
                    rule.name not in macros
                ), f"Macro {rule.name} cannot be defined more than once."
                macros[rule.name] = rule
            else:
                assert isinstance(rule, Rule), rule
                rules.append(rule)
        return PreprocessedGrammar.from_rules(rules=rules, macros=macros)

    @classmethod
    def from_rules(
        cls, rules: List[Rule], macros: Optional[Dict[str, Macro]] = None
    ) -> "PreprocessedGrammar":
        if macros is None:
            macros = {}
        sync_rules: Dict[Tuple[Nonterminal, Expansion], Set[Expansion]] = defaultdict(
            set
        )
        utterance_rules: Dict[Nonterminal, Set[Expansion]] = defaultdict(set)

        ###
        # Figure out if the rule is a plan rule, an utterance rule, or a sync rule.
        # For each rule with a plan rhs, rewrite it so all macros are expanded.
        #
        # if the rule is an utterance rule or a sync rule, with tokens that are optional (e.g. x? or "the"?)
        # compile the rule into multiple rules with no optional tokens.
        # TODO: It's confusing that `all_possible_options` outputs `Expansion`s that have
        #  `OptionableSCFGToken`s in them, when in reality after this point none of our code can
        #  handle `optional=True`. It all assumes that they have been compiled away by
        #  `expand_optionals`.
        ####
        for rule in rules:
            if isinstance(rule, SyncRule):
                rule = replace(rule, plan_rhs=expand_macros(macros, rule.plan_rhs))
            for new_rule in expand_optionals(rule):
                if isinstance(new_rule, UtteranceRule):
                    utterance_rules[new_rule.lhs] |= set(new_rule.utterance_rhss)
                elif isinstance(new_rule, SyncRule):
                    sync_rules[new_rule.lhs, new_rule.plan_rhs] |= set(
                        new_rule.utterance_rhss
                    )
            # TODO: we're throwing away plan_rules. weird?
        return PreprocessedGrammar(sync_rules, utterance_rules)

    @staticmethod
    def from_folder(folder_path: StrPath) -> "PreprocessedGrammar":
        all_scfg_paths = find_all_scfg_paths(folder_path)
        if not all_scfg_paths:
            raise ValueError(f"No .scfg files found under {folder_path!r}.")
        all_scfg_paths.sort()
        with contextlib.ExitStack() as stack:
            line_iters = [
                iter(stack.enter_context(BlobFile(filename, streaming=False)))
                for filename in all_scfg_paths
            ]
            return PreprocessedGrammar.from_line_iter(
                itertools.chain.from_iterable(line_iters)
            )
        # https://github.com/microsoft/pyright/issues/1514
        raise Exception("Exception inside ExitStack")
