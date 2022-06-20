# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from pathlib import Path

from semantic_parsing_with_constrained_lm.earley.cfg import load_grammar_from_directory
from semantic_parsing_with_constrained_lm.datum import DatumSub
from semantic_parsing_with_constrained_lm.decoding.earley_partial_parse import (
    GrammarTokenizerInfo,
    UTF8EarleyPartialParse,
)
from semantic_parsing_with_constrained_lm.decoding.partial_parse import StartsWithSpacePartialParse
from semantic_parsing_with_constrained_lm.domains.benchclamp_data_setup import (
    BenchClampDataset,
    BenchClampDatasetConfig,
)
from semantic_parsing_with_constrained_lm.domains.lispress_v2.grammar import (
    create_partial_parse_builder as create_partial_parse_builder_lispress_v2,
)
from semantic_parsing_with_constrained_lm.domains.mtop.grammar import (
    create_partial_parse_builder as create_partial_parse_builder_mtop,
)
from semantic_parsing_with_constrained_lm.domains.sql.cosql.grammar import (
    load_base_grammar,
    preprocessed_grammar_for_schema,
)
from semantic_parsing_with_constrained_lm.domains.sql.cosql.schema import load_schemas
from semantic_parsing_with_constrained_lm.model import PartialParseBuilder
from semantic_parsing_with_constrained_lm.paths import BENCH_CLAMP_GRAMMAR_DATA_DIR_AZURE
from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer

TEST_SUITE_PATH = Path("/mnt/my_input/test-suite-sql-eval")
TEST_SUITE_DATABASE_PATH = Path("/mnt/my_input/test-suite-sql-eval/database/")
SPIDER_DATABASE_PATH = Path("/mnt/my_input/Spider/database/")
SPIDER_TABLES_FILE = Path("/mnt/my_input/Spider/tables.json")
COSQL_DATABASE_PATH = Path("/mnt/my_input/CoSQL/database/")
COSQL_TABLES_FILE = Path("/mnt/my_input/CoSQL/tables.json")


def create_partial_parse_builder(
    constrained: bool, data_config: BenchClampDatasetConfig, tokenizer: ClampTokenizer
) -> PartialParseBuilder[DatumSub]:
    if constrained:
        domain_str = data_config.domain if data_config.domain is not None else ""
        if data_config.dataset_name in [
            BenchClampDataset.Spider.value,
            BenchClampDataset.CoSQL.value,
        ]:
            print("Loading database schemas ...")
            if data_config.dataset_name == BenchClampDataset.Spider.value:
                schemas = load_schemas(
                    schemas_path=SPIDER_TABLES_FILE,
                    db_path=SPIDER_DATABASE_PATH,
                )
            else:
                schemas = load_schemas(
                    schemas_path=COSQL_TABLES_FILE, db_path=COSQL_DATABASE_PATH
                )
            print("Done")

            base_grammar = load_base_grammar()
            pre_grammars = {
                name: preprocessed_grammar_for_schema(db, base_grammar)
                for name, db in schemas.items()
            }
            grammar_tok_info = {
                name: GrammarTokenizerInfo.create(tokenizer, preprocessed_grammar, True)
                for name, preprocessed_grammar in pre_grammars.items()
            }
            partial_parse_builder = lambda datum: UTF8EarleyPartialParse.initial(
                grammar_tok_info[datum.schema_name], datum.natural  # type: ignore
            )

        elif data_config.dataset_name in (
            BenchClampDataset.CalFlowV2.value,
            BenchClampDataset.TreeDST.value,
        ):
            partial_parse_builder = create_partial_parse_builder_lispress_v2(
                load_grammar_from_directory(
                    os.path.join(
                        BENCH_CLAMP_GRAMMAR_DATA_DIR_AZURE,
                        data_config.dataset_name,
                        domain_str,
                    )
                ),
                tokenizer,
            )
        elif data_config.dataset_name == BenchClampDataset.MTOP.value:
            partial_parse_builder = create_partial_parse_builder_mtop(
                load_grammar_from_directory(
                    os.path.join(
                        BENCH_CLAMP_GRAMMAR_DATA_DIR_AZURE,
                        data_config.dataset_name,
                        domain_str,
                    )
                ),
                tokenizer,
            )
        else:
            raise ValueError(f"{data_config.dataset_name} not supported")
    else:
        partial_parse = StartsWithSpacePartialParse(tokenizer)
        partial_parse_builder = lambda _: partial_parse
    return partial_parse_builder
