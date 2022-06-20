# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass

from openfst_python import determinize, intersect

from semantic_parsing_with_constrained_lm.earley.fsa import CompiledDFA
from semantic_parsing_with_constrained_lm.earley.fsa_builders import compile_nfa, re_substring_utf8
from semantic_parsing_with_constrained_lm.earley.grammar import DFADottedRule, DFAGrammar, Nonterm


@dataclass
class SubstringIntersectingGrammarSpecializer:
    """Creates grammars where the rule for string literals are modified to only allow substrings of a given string.

    The string is usually a user's utterance, where we are performing semantic parsing of that utterance
    and we know that any strings in the output program must be substrings of the utterance."""

    base_grammar: DFAGrammar
    nonterm_to_intersect: Nonterm

    def __post_init__(self):
        assert self.nonterm_to_intersect in self.base_grammar.expansions

    def specialize(self, s: str) -> DFAGrammar:
        existing_rule = self.base_grammar.expansions[self.nonterm_to_intersect]
        substring_nfa = compile_nfa(re_substring_utf8(s))
        assert substring_nfa.edge_indexer.num_indexed() == 0

        # Intersect the FSA which accepts any valid string literal with the FSA
        # for the substrings of `s`.  This ensures that we don't take any
        # substrings of `s` which are incorrectly escaped, e.g.  taking " rather
        # than \".
        # TODO: Have some checks that `s` has already been escaped correctly,
        # since otherwise we will end up excluding many substrings which should
        # have been included.
        new_dfa = CompiledDFA(
            determinize(
                intersect(existing_rule.dfa.fst, substring_nfa.fst).rmepsilon()
            ).minimize(),
            existing_rule.dfa.edge_indexer,
        )
        new_rule = DFADottedRule(existing_rule.lhs, new_dfa, new_dfa.start_id)

        return DFAGrammar(
            self.base_grammar.root,
            {**self.base_grammar.expansions, self.nonterm_to_intersect: new_rule},
        )
