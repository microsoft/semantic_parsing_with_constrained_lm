# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import functools

from semantic_parsing_with_constrained_lm.domains import dfa_grammar_utils

create_partial_parse_builder = functools.partial(
    dfa_grammar_utils.create_partial_parse_builder,
    utterance_nonterm_name="any_char_star",
)
