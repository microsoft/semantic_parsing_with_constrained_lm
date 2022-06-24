# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
from typing import List, Tuple

import pytest
import torch as t

from semantic_parsing_with_constrained_lm.scfg.char_grammar import NotParsable
from semantic_parsing_with_constrained_lm.scfg.generate import (
    parse_and_render,
    sample_from_grammar_and_nonterminal,
)
from semantic_parsing_with_constrained_lm.scfg.read_grammar import PreprocessedGrammar
from semantic_parsing_with_constrained_lm.scfg.scfg import SCFG
from semantic_parsing_with_constrained_lm.decoding.earley_partial_parse import (
    GrammarTokenizerInfo,
    UTF8EarleyPartialParse,
)
from semantic_parsing_with_constrained_lm.domains.sql.cosql.grammar import (
    grammar_for_schema,
    load_base_grammar,
    preprocessed_grammar_for_schema,
)
from semantic_parsing_with_constrained_lm.domains.sql.cosql.schema import (
    Column,
    ColumnType,
    DbSchema,
    Table,
)

WS = re.compile(r"\s+", re.MULTILINE)

ASCII_IDS = range(128)
ORDERED_IDS = t.LongTensor(ASCII_IDS)

INSTRUCTOR_SCHEMA = DbSchema(
    "instructor",
    [
        Table(
            "instructor",
            [
                Column("salary", ColumnType.Number),
                Column("dept_name", ColumnType.Text),
                Column("name", ColumnType.Text),
            ],
        ),
    ],
)


# some programs from CoSQL
UTTERANCES: List[Tuple[DbSchema, List[str]]] = [
    (
        INSTRUCTOR_SCHEMA,
        [
            "sElECT avg ( salary )  FROM instructor",
            "SELECT avg ( salary ) , dept_name FROM instructor GROUP BY dept_name",
            "SELECT dept_name FROM instructor GROUP BY dept_name ORDER BY avg ( salary )  DESC LIMIT 1",
            "SELECT dept_name FROM instructor GROUP BY dept_name ORDER BY avg ( salary )   LIMIT 1",
            "SELECT dept_name FROM instructor where name  =  'Mird'",
            "SELECT salary FROM instructor where name  =  'Mird'",
        ],
    ),
    (
        DbSchema(
            "trains",
            [
                Table(
                    "train_station",
                    [
                        Column("station_id", ColumnType.Number),
                        Column("train_id", ColumnType.Number),
                    ],
                ),
                Table(
                    "station",
                    [
                        Column("station_id", ColumnType.Number),
                        Column("name", ColumnType.Text),
                    ],
                ),
                Table("train", [Column("train_id", ColumnType.Number)]),
            ],
        ),
        [
            'SELECT station_id from station where name  =  "London Waterloo"',
            "SELECT * FROM train_station AS T1"
            " JOIN station AS T2 ON T1.station_id   =   T2.station_id"
            " JOIN train AS T3 ON T3.train_id   =   T1.train_id"
            " where T1.station_id  =  1",
        ],
    ),
    (
        DbSchema(
            "captain",
            [
                Table(
                    "captain",
                    [Column("rank", ColumnType.Text), Column("class", ColumnType.Text)],
                )
            ],
        ),
        ['SELECT Rank FROM captain where class ! =  "Third-rate ship of the line"'],
    ),
    (
        DbSchema(
            "film",
            [
                Table(
                    "track",
                    [
                        Column("milliseconds", ColumnType.Number),
                        Column("name", ColumnType.Text),
                    ],
                ),
                Table("film", [Column("studio", ColumnType.Text)]),
            ],
        ),
        [
            "SELECT max ( Milliseconds )  ,  min ( Milliseconds )  FROM TRACK",
            "SELECT distinct Studio FROM film",
            "SELECT name FROM TRACK where Milliseconds  =   ( select max ( Milliseconds )  from track )",
        ],
    ),
    (
        DbSchema(
            "customers",
            [
                Table(
                    "customers",
                    [
                        Column("customer_id", ColumnType.Number),
                        Column("customer_name", ColumnType.Text),
                    ],
                ),
                Table(
                    "orders",
                    [
                        Column("customer_id", ColumnType.Number),
                        Column("order_id", ColumnType.Number),
                    ],
                ),
                Table(
                    "order_items",
                    [
                        Column("order_id", ColumnType.Number),
                        Column("product_id", ColumnType.Number),
                        Column("order_item_status", ColumnType.Text),
                    ],
                ),
                Table(
                    "products",
                    [
                        Column("product_id", ColumnType.Number),
                        Column("product_name", ColumnType.Text),
                    ],
                ),
            ],
        ),
        [
            "SELECT T1.customer_name FROM customers AS T1"
            " JOIN orders AS T2 JOIN order_items AS T3"
            " JOIN products AS T4 ON T1.customer_id  =  T2.customer_id"
            " AND T2.order_id  =  T3.order_id"
            " AND T3.product_id  =  T4.product_id"
            ' WHERE T3.order_item_status  =  "Cancel"'
            ' AND T4.product_name  =  "food"'
            " GROUP BY T1.customer_id"
            " HAVING count ( * )    > =    1",
        ],
    ),
    (
        DbSchema(
            "college",
            [
                Table(
                    "college",
                    [
                        Column("state", ColumnType.Text),
                        Column("enr", ColumnType.Text),
                        Column("cname", ColumnType.Text),
                    ],
                ),
                Table(
                    "tryout",
                    [
                        Column("cname", ColumnType.Text),
                        Column("decision", ColumnType.Boolean),
                    ],
                ),
            ],
        ),
        [
            "select state, enr from college"
            " where cName not in  ("
            " SELECT DISTINCT T1.cName FROM college AS T1"
            " JOIN tryout AS T2 ON T1.cName   =   T2.cName"
            ' WHERE T2.decision  =  "yes"'
            " )",
        ],
    ),
    (
        DbSchema(
            "wine",
            [
                Table(
                    "wine",
                    [
                        Column("name", ColumnType.Text),
                        Column("price", ColumnType.Number),
                        Column("winery", ColumnType.Text),
                    ],
                ),
            ],
        ),
        [
            "SELECT DISTINCT Name FROM WINE"
            " WHERE Price  >   ("
            ' SELECT min ( Price )  FROM wine WHERE Winery   =   "John Anthony"'
            " )  and  Price>300",
        ],
    ),
    (
        DbSchema(
            "employees",
            [
                Table(
                    "employees",
                    [
                        Column("first_name", ColumnType.Text),
                        Column("last_name", ColumnType.Text),
                        Column("salary", ColumnType.Number),
                        Column("commission_pct", ColumnType.Number),
                    ],
                ),
            ],
        ),
        [
            "SELECT first_name,  salary FROM employees WHERE first_name NOT LIKE '%M%' ORDER BY salary DESC",
            "SELECT FIRST_NAME, LAST_NAME FROM employees order by COMMISSION_PCT desc limit 1",
        ],
    ),
    (
        DbSchema("address", [Table("address", [Column("address", ColumnType.Text)])]),
        ['SELECT address FROM address WHERE address LIKE "%S%"'],
    ),
    (
        DbSchema(
            "department_management",
            [
                Table(
                    "management",
                    [
                        Column("temporary_acting", ColumnType.Text),
                        Column("age", ColumnType.Number),
                        Column("head_id", ColumnType.Number),
                    ],
                ),
                Table(
                    "head",
                    [
                        Column("head_id", ColumnType.Number),
                        Column("name", ColumnType.Text),
                    ],
                ),
            ],
        ),
        [
            # # regression test for whitespace bug
            "SELECT count(DISTINCT temporary_acting) FROM management",
            # regression tests for `in(select`
            "SELECT count(*) FROM management WHERE head_id NOT IN(SELECT head_id FROM head)",
            "SELECT count(DISTINCT temporary_acting) FROM management"
            " WHERE head_id NOT IN( SELECT head_id FROM head WHERE name != 'blah' )",
            # # regression tests for force-decoding
            "SELECT DISTINCT T1 . age FROM management AS T2"
            " JOIN head AS T1 ON T1 . head_id = T2 . head_id"
            " WHERE T2 . temporary_acting = 'Yes'",
            "SELECT head_id , name FROM head WHERE name LIKE '%Ha%'",
        ],
    ),
    # regression test for what ended up being a schema error in Spider.
    # `company_type_code` is actually `company_type` in tables.json
    (
        DbSchema(
            "assets_maintenance",
            [
                Table(
                    "third_party_companies",
                    [
                        Column("company_id", ColumnType.Number),
                        Column("company_name", ColumnType.Text),
                        Column("company_type_code", ColumnType.Text),
                    ],
                ),
                Table(
                    "maintenance_contracts",
                    [
                        Column("maintenance_contract_company_id", ColumnType.Text),
                        Column("contract_end_date", ColumnType.Time),
                    ],
                ),
                Table(
                    "ref_company_types", [Column("company_type_code", ColumnType.Text)]
                ),
            ],
        ),
        [
            "SELECT T1.company_name FROM Third_Party_Companies AS T1"
            ""
            " JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id"
            " JOIN Ref_Company_Types AS T3 ON T1.company_type_code  =  T3.company_type_code"
            " ORDER BY T2.contract_end_date DESC LIMIT 1",
        ],
    ),
    # # These are in CoSQL, but invalid
    # 'SELECT avg ( T2.rating )  FROM useracct'
    # ' WHERE T2.u_id = 1 AS T1 JOIN review AS T2 ON T1.u_id   =   T2.u_id GROUP BY T2.u_id',
    # "select Claim_ID from Claims_Processing_Stages AS T1"
    # " JOIN Claims_Processing AS T2 ON T1.Claim_Stage_ID  =  T2.Claim_Stage_ID"
    # " WHEREClaim_Status_Name = 'Open'",
    # 'SELECT order_id,  from Order_Items group by order_id order by sum ( order_quantity )  desc limit 1',
    # # Spider schema: document_management
    # # Error: in prepare, ORDER BY clause should come after INTERSECT not before (1)
    # "SELECT document_name FROM documents"
    # " GROUP BY document_type_code ORDER BY count(*) DESC LIMIT 3"
    # " INTERSECT SELECT document_name FROM documents"
    # " GROUP BY document_structure_code ORDER BY count(*) DESC LIMIT 3",
    # # here are corrected versions
    (
        DbSchema(
            "useracct",
            [
                Table("useracct", [Column("u_id", ColumnType.Number)]),
                Table(
                    "review",
                    [
                        Column("rating", ColumnType.Number),
                        Column("u_id", ColumnType.Number),
                    ],
                ),
            ],
        ),
        [
            "SELECT avg ( T2.rating ) FROM useracct AS T1"
            " JOIN review AS T2 ON T1.u_id   =   T2.u_id"
            " WHERE T2.u_id = 1 GROUP BY T2.u_id",
        ],
    ),
    (
        DbSchema(
            "claims",
            [
                Table(
                    "claims_processing_stages",
                    [
                        Column("claim_id", ColumnType.Number),
                        Column("claim_stage_id", ColumnType.Number),
                    ],
                ),
                Table(
                    "claims_processing",
                    [
                        Column("claim_stage_id", ColumnType.Number),
                        Column("claim_status_name", ColumnType.Text),
                    ],
                ),
            ],
        ),
        [
            "select Claim_ID from Claims_Processing_Stages AS T1"
            " JOIN Claims_Processing AS T2 ON T1.Claim_Stage_ID  =  T2.Claim_Stage_ID"
            " WHERE Claim_Status_Name = 'Open'",
        ],
    ),
    (
        DbSchema(
            "orders",
            [
                Table(
                    "Order_Items",
                    [
                        Column("order_id", ColumnType.Number),
                        Column("order_quantity", ColumnType.Number),
                    ],
                )
            ],
        ),
        [
            "SELECT order_id  from Order_Items group by order_id order by sum ( order_quantity )  desc limit 1",
        ],
    ),
]


def upper_rm_space(s: str) -> str:
    return WS.sub("", s).upper()


def make_info(grammar: PreprocessedGrammar) -> GrammarTokenizerInfo:
    vocab = [chr(i) for i in ASCII_IDS]  # all ascii chars are tokens
    return GrammarTokenizerInfo.from_tokens_list(vocab, grammar, for_plans=True)


def is_grammatical_prefix(
    prefix: str, input_utterance: str, info: GrammarTokenizerInfo
) -> bool:
    start = UTF8EarleyPartialParse.initial(info, input_utterance)
    curr = start
    for c in prefix:
        try:
            curr = curr.append(ord(c))
        except KeyError:
            return False
    allowed_next_ids, _ = curr.allowed_next(ORDERED_IDS)
    if allowed_next_ids is None:
        return False
    allowed_next_chars = [chr(i) for i in allowed_next_ids.tolist()]
    # print(allowed_next_chars)
    return len(allowed_next_chars) > 0


class TestSqlGrammar:
    @staticmethod
    @pytest.fixture(scope="class")
    def base_sql_grammar() -> PreprocessedGrammar:
        return load_base_grammar()

    @staticmethod
    @pytest.fixture(scope="class")
    def grammar(base_sql_grammar: PreprocessedGrammar) -> PreprocessedGrammar:
        (schema, _), *_ = UTTERANCES
        return preprocessed_grammar_for_schema(schema, base_sql_grammar)

    @staticmethod
    @pytest.fixture(scope="class")
    def info(grammar: PreprocessedGrammar) -> GrammarTokenizerInfo:
        return make_info(grammar)

    @staticmethod
    def test_real_examples(base_sql_grammar: PreprocessedGrammar):
        for schema, utterances in UTTERANCES:
            # print(schema.name)
            grammar = grammar_for_schema(schema, base_sql_grammar)
            for utterance in utterances:
                # print("utt ", utterance)
                plan = next(
                    parse_and_render(
                        grammar, utterance, source_is_plan=False, max_depth=500
                    )
                )
                # print("plan", plan)
                # should normalize whitespace seqs to a single space
                assert "  " not in plan
                # should not change anything except whitespace and capitalization
                assert upper_rm_space(utterance) == upper_rm_space(plan)
                # should be parsable in the reverse direction
                _round_tripped = next(
                    parse_and_render(grammar, plan, source_is_plan=True, max_depth=500)
                )

    @staticmethod
    def test_no_repeated_nes(base_sql_grammar: PreprocessedGrammar):
        # LMs sometimes like to predict junk like this.
        # It's questionable whether it's valid SQL, so we disallow it in the grammar.
        instructor_grammar = preprocessed_grammar_for_schema(
            INSTRUCTOR_SCHEMA, base_sql_grammar
        )
        instructor_info = make_info(instructor_grammar)
        instructor_sql = (
            " SELECT count ( * ) FROM instructor"
            ' ( DISTINCT ( name ) != 1 != "n" != "n" != "n"'
        )
        assert not is_grammatical_prefix(instructor_sql, "n", instructor_info)
        scfg = SCFG(instructor_grammar)
        parses = parse_and_render(scfg, instructor_sql, False)
        assert isinstance(parses, NotParsable)
        assert parses.longest_parsable_prefix == " SELECT count ( * ) FROM instructor "
        assert parses.expected_next_terminals.issuperset(
            {" ", "W", "J"}
        )  # "WHERE", "JOIN"
        assert parses.actual_next_char == "("
        assert parses.actual_next_char not in parses.expected_next_terminals

        assert is_grammatical_prefix(
            " SELECT count ( * ) FROM instructor", "", instructor_info
        )

        # This is valid SQL, but we disallow it in the grammar.
        tv_schema = DbSchema(
            "tv", [Table("tv_channel", [Column("language", ColumnType.Text)])]
        )
        tv_info = make_info(
            preprocessed_grammar_for_schema(tv_schema, base_sql_grammar)
        )
        tv_sql = (
            " SELECT count ( * ) FROM tv_channel"
            " WHERE language = 'English'"
            " EXCEPT SELECT count ( * ) FROM tv_channel"
            ' WHERE language = \'English\' != "English" != "English" != "English" !='
        )
        assert not is_grammatical_prefix(tv_sql, "English", tv_info)

    @staticmethod
    def test_no_weird_agg_funcs(info: GrammarTokenizerInfo):
        for agg_func in ["max", "min", "count", "sum", "avg"]:
            assert is_grammatical_prefix(
                f" SELECT {agg_func} ( * ) FROM instructor", "", info
            )
        assert not is_grammatical_prefix(
            " SELECT asdf ( * ) FROM instructor", "asdf", info
        )

    @staticmethod
    @pytest.mark.skip(reason="slow")
    def test_generate(grammar: PreprocessedGrammar):
        """Tests that non-trivial utterances generated by the grammar can be parsed by the grammar."""
        scfg = SCFG(grammar)
        count = 0
        while count < 10:
            try:
                utterance = sample_from_grammar_and_nonterminal(scfg).render()
                plan = next(
                    parse_and_render(
                        scfg, utterance, source_is_plan=False, max_depth=1000
                    )
                )
                assert upper_rm_space(utterance) == upper_rm_space(plan)
                if WS.sub("", plan).replace(";", "") == "":
                    # skip trivial examples to make sure we get enough non-trivial tests
                    continue
                # print()
                # print("utt: ", WS.sub(' ', utterance))
                # print("plan:", plan)
                count += 1
            except RecursionError:
                continue
