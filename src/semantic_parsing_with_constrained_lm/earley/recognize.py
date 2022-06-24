# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Methods that illustrates the simplest ways to use EarleyChart:
as a recognizer of a token string, or as a generator of grammatical sentences.
"""

from typing import Iterable, Iterator, List, Sequence, cast

from semantic_parsing_with_constrained_lm.earley.earley import EarleyLRChart, PackedForest
from semantic_parsing_with_constrained_lm.earley.grammar import Grammar, RuleResult, Terminal
from semantic_parsing_with_constrained_lm.earley.input import SequencePosition, SigmaStarTriePosition


def parse(
    sentence: Iterable[Terminal], grammar: Grammar[Terminal, RuleResult]
) -> PackedForest[Terminal]:
    start_pos = SequencePosition(list(sentence))
    chart = EarleyLRChart(grammar=grammar, start_pos=start_pos, use_backpointers=True)
    return chart.parse()


def is_grammatical(
    tokens: Sequence[Terminal], grammar: Grammar[Terminal, RuleResult]
) -> bool:
    """
    Tests whether the given input `tokens` are grammatical under `grammar`.
    """
    start_pos = SequencePosition(tokens)
    chart = EarleyLRChart(grammar, start_pos, use_backpointers=False)
    for _ in chart.accepting_positions():
        return True  # we're grammatical if the iterator is non-empty
    return False


def top_level_rule_results(
    tokens: Sequence[Terminal], grammar: Grammar[Terminal, RuleResult]
) -> Iterable[RuleResult]:
    """
    Yields the RuleResults produced by the DottedRules from the `start` nonterminal.
    """
    start_pos = SequencePosition(tokens)
    chart = EarleyLRChart(grammar, start_pos, use_backpointers=False)
    for end_pos in chart.accepting_positions():
        for item, _ in chart.completed_items(grammar.root, start_pos, end_pos):
            rule_result = item.dotted_rule.is_final()
            assert rule_result is not None
            yield rule_result


def enumerate_sentences(
    grammar: Grammar[Terminal, RuleResult]
) -> Iterator[List[Terminal]]:
    """
    Yields grammatical sentences in length order (may not terminate).
    """
    # root of a Σ* trie with string-labeled edges (as the grammar uses Terminal=str)
    start_pos = SigmaStarTriePosition[Terminal]()
    chart = EarleyLRChart(grammar, start_pos, use_backpointers=False)
    for pos in chart.accepting_positions():  # enumerate nodes in the Σ* trie
        # necessary because current typing isn't strong enough
        _pos = cast(SigmaStarTriePosition[Terminal], pos)
        yield _pos.prefix()
