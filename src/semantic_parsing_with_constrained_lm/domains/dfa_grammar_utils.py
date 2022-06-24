# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from semantic_parsing_with_constrained_lm.earley.grammar import DFAGrammar, Nonterm
from semantic_parsing_with_constrained_lm.earley.specialization import SubstringIntersectingGrammarSpecializer
from semantic_parsing_with_constrained_lm.datum import Datum
from semantic_parsing_with_constrained_lm.decoding.partial_parse import PartialParse
from semantic_parsing_with_constrained_lm.decoding.uint8_earley_partial_parse import (
    UInt8EarleyPartialParse,
    UInt8GrammarTokenizerInfo,
)
from semantic_parsing_with_constrained_lm.model import PartialParseBuilder
from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer


def create_partial_parse_builder(
    grammar: DFAGrammar, tokenizer: ClampTokenizer, utterance_nonterm_name: str
) -> PartialParseBuilder[Datum]:
    specializer = SubstringIntersectingGrammarSpecializer(
        grammar, Nonterm(utterance_nonterm_name)
    )
    tokens = UInt8GrammarTokenizerInfo.prepare_tokens_from_clamp_tokenizer(tokenizer)

    def builder(datum: Datum) -> PartialParse:
        specialized_grammar = specializer.specialize(datum.natural)
        return UInt8EarleyPartialParse.initial(
            UInt8GrammarTokenizerInfo(specialized_grammar, tokens)
        )

    return builder
