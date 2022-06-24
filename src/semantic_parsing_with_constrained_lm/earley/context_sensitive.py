# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Tools to help create context-sensitive grammars.
# See tests/test_harbor/earley/context_sensitive_demos for some example uses.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, Iterable, Set

from semantic_parsing_with_constrained_lm.earley.grammar import DottedRule, Nonterm, RuleResult, Terminal


class SelfExpandingNonterm(Generic[Terminal, RuleResult], Nonterm, ABC):
    @abstractmethod
    def get_expansions(self) -> Iterable[DottedRule[Terminal, RuleResult]]:
        """Get the set of expansions for this nonterminal.

        Child classes should contain additional fields which are used to
        determine the expansions created."""
        pass


@dataclass
class DynamicGrammar(Generic[Terminal, RuleResult]):
    """A Grammar which knows how to use SelfExpandingNonterm to create expansions."""

    root: Nonterm
    fixed_expansions: Dict[Nonterm, Set[DottedRule[Terminal, RuleResult]]]

    def get_expansions(
        self, nonterm: Nonterm
    ) -> Iterable[DottedRule[Terminal, RuleResult]]:
        result = self.fixed_expansions.get(nonterm)
        if result is not None:
            return result

        if isinstance(nonterm, SelfExpandingNonterm):
            return nonterm.get_expansions()

        return ()
