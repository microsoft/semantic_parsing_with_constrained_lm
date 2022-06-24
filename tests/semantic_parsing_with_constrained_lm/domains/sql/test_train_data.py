# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from semantic_parsing_with_constrained_lm.scfg.generate import parse_and_render
from semantic_parsing_with_constrained_lm.domains.sql.cosql.dialogue import load_cosql_data
from semantic_parsing_with_constrained_lm.domains.sql.cosql.grammar import (
    grammar_for_schema,
    load_base_grammar,
)
from semantic_parsing_with_constrained_lm.domains.sql.cosql.schema import load_schemas


@pytest.mark.skip(reason="slow")
class TestTrainData:
    @staticmethod
    def test_all_queries_parse():
        base_grammar = load_base_grammar()
        schemas = load_schemas()
        dialogues = list(load_cosql_data())
        for i, dialogue in enumerate(dialogues):
            schema = schemas[dialogue.schema_name]
            grammar = grammar_for_schema(schema, base_grammar)
            print()
            print(i, dialogue.schema_name)
            for turn in dialogue.interaction + [dialogue.final]:
                try:
                    parsed = next(parse_and_render(grammar, turn.query, False))
                except StopIteration:
                    print()
                    print("-------------------------")
                    print("FAILED TO TO PARSE QUERY:")
                    print(turn.query)
                    print("-------------------------")
                    print()
                else:
                    print(parsed)
