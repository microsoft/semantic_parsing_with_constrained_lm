# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from dataclasses import dataclass
from typing import Any, Dict, List

import tqdm

from semantic_parsing_with_constrained_lm.util.types import StrPath
from semantic_parsing_with_constrained_lm.scfg.generate import parse_and_render
from semantic_parsing_with_constrained_lm.datum import BenchClampDatum
from semantic_parsing_with_constrained_lm.domains.sql.cosql.paths import TRAIN_DATA_FILE
from semantic_parsing_with_constrained_lm.domains.sql.cosql.schema import DbSchema
from semantic_parsing_with_constrained_lm.domains.sql.cosql.seq2seq import serialize_schema


@dataclass(frozen=True)
class CoSqlTurn:
    utterances: List[str]
    query: str

    @staticmethod
    def from_json(turn_json: Dict[str, Any]) -> "CoSqlTurn":
        return CoSqlTurn(
            utterances=turn_json["utterance"].split(sep="|"), query=turn_json["query"]
        )


@dataclass(frozen=True)
class CoSqlDialogue:
    # index into `DbSchema.name`
    schema_name: str
    final: CoSqlTurn
    interaction: List[CoSqlTurn]

    @staticmethod
    def from_json(dialogue_json: Dict[str, Any]) -> "CoSqlDialogue":
        return CoSqlDialogue(
            schema_name=dialogue_json["database_id"],
            final=CoSqlTurn.from_json(dialogue_json["final"]),
            interaction=[
                CoSqlTurn.from_json(turn_json)
                for turn_json in dialogue_json["interaction"]
            ],
        )


@dataclass(frozen=True)
class SpiderDatum:
    schema_name: str
    query: str
    utterance: str

    @staticmethod
    def from_json(datum_json: Dict[str, Any]) -> "SpiderDatum":
        return SpiderDatum(
            schema_name=datum_json["db_id"],
            utterance=datum_json["question"],
            query=datum_json["query"],
        )


def load_cosql_data(data_filepath: StrPath = TRAIN_DATA_FILE) -> List[CoSqlDialogue]:
    with open(data_filepath, encoding="utf-8") as f:
        json_data = json.load(f)
        return [CoSqlDialogue.from_json(d) for d in json_data]


def convert_cosql_to_datum_format(
    cosql_dialogues: List[CoSqlDialogue], db_map: Dict[str, DbSchema], db_path: str
) -> List[BenchClampDatum]:
    data = []
    for dialogue_index, dialogue in tqdm.tqdm(enumerate(cosql_dialogues)):
        db_schema = db_map[dialogue.schema_name]
        table_names = [table.name for table in db_schema.tables]
        column_names = {
            "table_id": [table_id for table_id, _ in db_schema.columns],
            "column_name": [column.name for _, column in db_schema.columns],
        }
        utterances_till_now = []
        for turn_index, turn in enumerate(dialogue.interaction):
            utterances_till_now.extend(turn.utterances)
            utterance_with_context = " | ".join(utterances_till_now)
            serialized_schema_without_val = serialize_schema(
                utterance_with_context,
                db_path=db_path,
                db_id=dialogue.schema_name,
                db_column_names=column_names,
                db_table_names=table_names,
                schema_serialization_with_db_content=False,
            )
            serialized_schema_with_val = serialize_schema(
                utterance_with_context,
                db_path=db_path,
                db_id=dialogue.schema_name,
                db_column_names=column_names,
                db_table_names=table_names,
                schema_serialization_with_db_content=True,
            )
            datum = BenchClampDatum(
                dialogue_id=str(dialogue_index),
                turn_part_index=turn_index,
                utterance=utterance_with_context,
                plan=turn.query,
                schema_name=dialogue.schema_name,
                db_schema_without_val=serialized_schema_without_val,
                db_schema_with_val=serialized_schema_with_val,
            )
            data.append(datum)

    return data


def canonicalize_sql_with_grammar(sql: str, grammar) -> str:
    try:
        canonicalized_sql = next(parse_and_render(grammar, " " + sql, False))[1:]
    except:
        print(f"Could not parse {sql}")
        return sql
    return canonicalized_sql


def load_spider_data(data_filepath: StrPath = TRAIN_DATA_FILE) -> List[SpiderDatum]:
    with open(data_filepath, encoding="utf-8") as f:
        json_data = json.load(f)
        return [SpiderDatum.from_json(d) for d in json_data]


def convert_spider_to_datum_format(
    spider_data: List[SpiderDatum], db_map: Dict[str, DbSchema], db_path: str
) -> List[BenchClampDatum]:
    data = []
    for index, spider_datum in tqdm.tqdm(enumerate(spider_data)):
        db_schema = db_map[spider_datum.schema_name]
        table_names = [table.name for table in db_schema.tables]
        column_names = {
            "table_id": [table_id for table_id, _ in db_schema.columns],
            "column_name": [column.name for _, column in db_schema.columns],
        }
        serialized_schema_without_val = serialize_schema(
            spider_datum.utterance,
            db_path=db_path,
            db_id=spider_datum.schema_name,
            db_column_names=column_names,
            db_table_names=table_names,
            schema_serialization_with_db_content=False,
        )
        serialized_schema_with_val = serialize_schema(
            spider_datum.utterance,
            db_path=db_path,
            db_id=spider_datum.schema_name,
            db_column_names=column_names,
            db_table_names=table_names,
            schema_serialization_with_db_content=True,
        )
        datum = BenchClampDatum(
            dialogue_id=str(index),
            turn_part_index=0,
            utterance=spider_datum.utterance,
            plan=spider_datum.query,
            schema_name=spider_datum.schema_name,
            db_schema_without_val=serialized_schema_without_val,
            db_schema_with_val=serialized_schema_with_val,
        )
        data.append(datum)

    return data


def spider_get_input(question: str, serialized_schema: str) -> str:
    return question.strip() + " " + serialized_schema.strip()


def cosql_get_input(
    utterances: List[str], serialized_schema: str, sep: str = " | "
) -> str:
    # "[prefix] [utterance n] [serialized schema] || [utterance n-1] | [utterance n-2] | ..."
    if len(utterances) > 1:
        reversed_utterance_head = (
            utterance.strip() for utterance in reversed(utterances[:-1])
        )
        serialized_reversed_utterance_head = " || " + sep.join(reversed_utterance_head)
    else:
        serialized_reversed_utterance_head = ""
    return (
        utterances[-1].strip()
        + " "
        + serialized_schema.strip()
        + serialized_reversed_utterance_head
    )
