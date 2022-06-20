# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Iterator, List

from semantic_parsing_with_constrained_lm.util.types import StrPath
from semantic_parsing_with_constrained_lm.scfg.parser.rule import (
    Rule,
    SyncRule,
    mirrored_rule,
    nonterm,
    term,
)
from semantic_parsing_with_constrained_lm.scfg.read_grammar import PreprocessedGrammar
from semantic_parsing_with_constrained_lm.scfg.scfg import SCFG
from semantic_parsing_with_constrained_lm.domains.sql.cosql.paths import SQL_GRAMMAR_DIR
from semantic_parsing_with_constrained_lm.domains.sql.cosql.schema import DbSchema

# NTs in the SQL scfg
SCHEMA_NAME = "schema_name"
TABLE_NAME = "table_name"
TABLE_ALIAS = "table_alias"
COLUMN_NAME = "column_name"
POSSIBLY_QUALIFIED = "possibly_qualified_column_name"
DOT = "DOT"
WS_STAR = "ws_star"
SCHEMA_DOT_WS = "schema_dot_ws"


def export_sync_rules(db: DbSchema) -> Iterator[SyncRule]:
    """
    Exports schema-specific SCFG rules for the NTs
    `schema`, `table_name`, `column_name`, and `possibly_qualified_column_name`.
    Ensures that `(({s}.)?{t}.)?{c}` is accepted iff `c` is a column in table `t` in schema `s`.
    """
    # helpers
    def to_lower(lhs: str, rhs: str):
        """case-insensitive match, normalize to lowercase"""
        return SyncRule(
            lhs=lhs,
            # case-insensitive
            utterance_rhss=(
                tuple(nonterm(c.upper()) if c.isalpha() else term(c) for c in rhs),
            ),
            # return lower case
            plan_rhs=(term(rhs.lower()),),
        )

    rhs = (term(db.name),)
    yield mirrored_rule(SCHEMA_NAME, rhs)
    for table in db.tables:
        tbl_name = table.name
        my_tbl_name_nt = f"table_called_{tbl_name}"
        yield mirrored_rule(TABLE_NAME, (nonterm(my_tbl_name_nt),))
        # case-insensitive, normalize to lowercase
        yield to_lower(lhs=my_tbl_name_nt, rhs=tbl_name)
        # allow table alias
        yield mirrored_rule(my_tbl_name_nt, (nonterm(TABLE_ALIAS),))
        qualifier = f"qualifier_for_{tbl_name}"
        rhs3 = (
            nonterm(SCHEMA_DOT_WS, optional=True),
            nonterm(my_tbl_name_nt),
            nonterm(WS_STAR),
            nonterm(DOT),
            nonterm(WS_STAR),
        )
        yield mirrored_rule(qualifier, rhs3)
        col_nt = f"column_for_{tbl_name}"
        yield mirrored_rule(COLUMN_NAME, ((nonterm(col_nt)),))
        # ensures that `table.column` is accepted iff column is in table
        poss_qual_rhs = (nonterm(qualifier, optional=True), (nonterm(col_nt)))
        yield mirrored_rule(POSSIBLY_QUALIFIED, poss_qual_rhs)
        for column in table.all_columns():
            col_name = column.name
            # case-insensitive, normalize to lowercase
            yield to_lower(lhs=col_nt, rhs=col_name)

    # This will allow copying from database values.
    for val in db.values:
        t_val = (term(val),)
        yield mirrored_rule("non_single_quote_star", t_val)
        yield mirrored_rule("non_double_quote_star", t_val)


def load_base_grammar(folder_path: StrPath = SQL_GRAMMAR_DIR) -> PreprocessedGrammar:
    return PreprocessedGrammar.from_folder(folder_path)


def preprocessed_grammar_for_schema(
    db: DbSchema, base_grammar: PreprocessedGrammar
) -> PreprocessedGrammar:
    rules: List[Rule] = list(export_sync_rules(db))
    return base_grammar.merge(PreprocessedGrammar.from_rules(rules))


def grammar_for_schema(db: DbSchema, base_grammar: PreprocessedGrammar) -> SCFG:
    return SCFG(preprocessed_grammar_for_schema(db, base_grammar))
