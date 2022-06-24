# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
from pathlib import Path
from typing import Dict, List

import tqdm
from transformers import GPT2Tokenizer

from semantic_parsing_with_constrained_lm.scfg.scfg import SCFG
from semantic_parsing_with_constrained_lm.datum import BenchClampDatum
from semantic_parsing_with_constrained_lm.decoding.earley_partial_parse import (
    GrammarTokenizerInfo,
    UTF8EarleyPartialParse,
)
from semantic_parsing_with_constrained_lm.domains.benchclamp_data_setup import BenchClampDataset
from semantic_parsing_with_constrained_lm.domains.create_benchclamp_splits import (
    can_force_decode,
    create_benchclamp_splits,
)
from semantic_parsing_with_constrained_lm.domains.sql.cosql.dialogue import (
    canonicalize_sql_with_grammar,
    convert_cosql_to_datum_format,
    convert_spider_to_datum_format,
    load_cosql_data,
    load_spider_data,
)
from semantic_parsing_with_constrained_lm.domains.sql.cosql.grammar import (
    load_base_grammar,
    preprocessed_grammar_for_schema,
)
from semantic_parsing_with_constrained_lm.domains.sql.cosql.schema import DbSchema, load_schemas
from semantic_parsing_with_constrained_lm.domains.sql.sql_datum import SqlDatum
from semantic_parsing_with_constrained_lm.paths import (
    BENCH_CLAMP_PROCESSED_DATA_DIR,
    BENCH_CLAMP_RAW_DATA_DIR,
)
from semantic_parsing_with_constrained_lm.tokenization import GPT2ClampTokenizer


def write_data_and_test_grammar(
    train_data: List[BenchClampDatum],
    dev_data: List[BenchClampDatum],
    schemas: Dict[str, DbSchema],
    datum_output_dir: Path,
) -> None:
    base_grammar = load_base_grammar()
    pre_grammars = {
        name: preprocessed_grammar_for_schema(db, base_grammar)
        for name, db in schemas.items()
    }
    grammars = {name: SCFG(pg) for name, pg in pre_grammars.items()}
    train_data_with_canonical_sql: List[BenchClampDatum] = []
    dev_data_with_canonical_sql: List[BenchClampDatum] = []
    print("Canonicalizing SQL ...")
    for data, data_with_canonical_sql in [
        (train_data, train_data_with_canonical_sql),
        (dev_data, dev_data_with_canonical_sql),
    ]:
        for datum in tqdm.tqdm(data):
            grammar = grammars[datum.schema_name]
            data_with_canonical_sql.append(
                dataclasses.replace(
                    datum, plan=canonicalize_sql_with_grammar(datum.plan, grammar)
                )
            )

    # Create data splits
    print("Creating data splits ...")
    create_benchclamp_splits(
        train_data_with_canonical_sql,
        dev_data_with_canonical_sql,
        None,
        datum_output_dir,
    )

    print("Testing ...")
    clamp_tokenizer = GPT2ClampTokenizer(GPT2Tokenizer.from_pretrained("gpt2"))
    grammar_tok_info = {
        name: GrammarTokenizerInfo.create(clamp_tokenizer, preprocessed_grammar, True)
        for name, preprocessed_grammar in pre_grammars.items()
    }
    partial_parse_builder = lambda datum: UTF8EarleyPartialParse.initial(
        grammar_tok_info[datum.schema_name], datum.natural
    )
    total = 0
    wrong = 0
    print("Testing if force decoding possible for first 100 examples")
    for datum in train_data_with_canonical_sql:
        if total >= 100:
            break
        total += 1
        if not can_force_decode(
            clamp_tokenizer,
            partial_parse_builder,
            SqlDatum(
                natural=datum.utterance,
                canonical=datum.plan,
                dialogue_id="",
                turn_part_index=0,
                agent_context="",
                schema_name=datum.schema_name,  # type: ignore
            ),
        ):
            print(f"Error: {datum.plan}")
            print(f"Schema: {datum.schema_name}")
            print(f"Utterance: {datum.utterance}")
            print()
            wrong += 1

    print(f"Force Decode Errors %: {wrong} / {total}")


def main():
    raw_spider_dir = BENCH_CLAMP_RAW_DATA_DIR / BenchClampDataset.Spider.value
    spider_schemas = load_schemas(
        schemas_path=raw_spider_dir / "tables.json", db_path=raw_spider_dir / "database"
    )
    spider_train, train_others, spider_dev = [
        convert_spider_to_datum_format(
            load_spider_data(raw_spider_dir / fn),
            db_map=spider_schemas,
            db_path=str(raw_spider_dir / "database"),
        )
        for fn in ["train_spider.json", "train_others.json", "dev.json"]
    ]
    spider_train.extend(
        [
            dataclasses.replace(datum, dialogue_id=f"other-{datum.dialogue_id}")
            for datum in train_others
        ]
    )
    write_data_and_test_grammar(
        train_data=spider_train,
        dev_data=spider_dev,
        schemas=spider_schemas,
        datum_output_dir=BENCH_CLAMP_PROCESSED_DATA_DIR
        / BenchClampDataset.Spider.value,
    )

    raw_cosql_dir = BENCH_CLAMP_RAW_DATA_DIR / BenchClampDataset.CoSQL.value
    cosql_schemas = load_schemas(
        schemas_path=raw_cosql_dir / "tables.json", db_path=raw_cosql_dir / "database"
    )
    cosql_train, cosql_dev = [
        convert_cosql_to_datum_format(
            load_cosql_data(raw_cosql_dir / "sql_state_tracking" / fn),
            db_map=cosql_schemas,
            db_path=str(raw_cosql_dir / "database"),
        )
        for fn in ["cosql_train.json", "cosql_dev.json"]
    ]
    write_data_and_test_grammar(
        train_data=cosql_train,
        dev_data=cosql_dev,
        schemas=cosql_schemas,
        datum_output_dir=BENCH_CLAMP_PROCESSED_DATA_DIR / BenchClampDataset.CoSQL.value,
    )


if __name__ == "__main__":
    main()
