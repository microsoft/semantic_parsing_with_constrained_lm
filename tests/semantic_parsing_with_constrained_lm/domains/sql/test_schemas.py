# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from semantic_parsing_with_constrained_lm.scfg.char_grammar import NotParsable
from semantic_parsing_with_constrained_lm.scfg.generate import parse_and_render
from semantic_parsing_with_constrained_lm.scfg.scfg import SCFG
from semantic_parsing_with_constrained_lm.domains.sql.cosql.grammar import (
    grammar_for_schema,
    load_base_grammar,
)
from semantic_parsing_with_constrained_lm.domains.sql.cosql.schema import (
    Column,
    ColumnType,
    DbSchema,
    Table,
)

TRAVEL_AGENT_SCHEMA = DbSchema(
    "travel_agent",
    [
        Table("city", [Column("city_id", ColumnType.Number)]),
        Table("customer", [Column("customer_id", ColumnType.Number)]),
    ],
)


class TestSchemas:
    @staticmethod
    @pytest.fixture(scope="class")
    def grammar() -> SCFG:
        return grammar_for_schema(TRAVEL_AGENT_SCHEMA, load_base_grammar())

    def test_matching_table_column_parses(self, grammar: SCFG):
        s = " SELECT travel_agent. city . City_ID  FROM city"
        parses = parse_and_render(grammar, s, source_is_plan=False)
        assert next(parses) == " SELECT travel_agent . city . city_id FROM city"

    def test_unmatched_table_column_doesnt_parse(self, grammar: SCFG):
        s = " SELECT travel_agent. city . Customer_ID  FROM city"
        parses = parse_and_render(grammar, s, source_is_plan=False)
        assert isinstance(parses, NotParsable)
        assert parses.longest_parsable_prefix == " SELECT travel_agent. city . C"
        assert parses.expected_next_terminals == {"I", "i"}
        assert parses.actual_next_char == "u"
        assert list(parses) == []
