# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from json import load
from typing import Any, Dict, List, Tuple

import jsons

from semantic_parsing_with_constrained_lm.util.types import StrPath
from semantic_parsing_with_constrained_lm.domains.sql.cosql.content_encoder import get_column_picklist
from semantic_parsing_with_constrained_lm.domains.sql.cosql.paths import COSQL_DIR, SCHEMAS_FILE


class ColumnType(Enum):
    Text = "text"
    Number = "number"
    Time = "time"
    Boolean = "boolean"
    Others = "others"


@dataclass(frozen=True)
class Column:
    name: str
    tpe: ColumnType
    # name in more natural language
    nl_name: str = ""

    @staticmethod
    def star() -> "Column":
        return Column(name="*", tpe=ColumnType.Text, nl_name="*")


@dataclass(frozen=True)
class ForeignKey:
    column_id: int
    other_column_id: int


@dataclass(frozen=True)
class Table:
    name: str
    columns: List[Column]
    # name in more natural language
    nl_name: str = ""

    def all_columns(self) -> List[Column]:
        return [Column.star()] + self.columns


@dataclass(frozen=True)
class DbSchema:
    name: str
    tables: List[Table]
    # indexes into self.tables
    columns: List[Tuple[int, Column]] = ()  # type: ignore
    # indexes into self.columns
    primary_keys: List[int] = ()  # type: ignore
    # indexes into self.columns
    foreign_keys: List[ForeignKey] = ()  # type: ignore
    # values in the database
    values: List[str] = ()  # type: ignore

    @staticmethod
    def from_json(schema_json: Dict[str, Any], db_path: str) -> "DbSchema":
        columns: List[Tuple[int, Column]] = [
            (t_id, Column(name=orig, tpe=ColumnType(tpe), nl_name=name))
            for (t_id, orig), (_, name), tpe in zip(
                schema_json["column_names_original"],
                schema_json["column_names"],
                schema_json["column_types"],
            )
        ]
        columns_by_table: Dict[int, List[Column]] = defaultdict(list)
        for t_id, col in columns:
            columns_by_table[t_id].append(col)
        # TODO:
        # tables.json is corrupted in the CoSQL dataset for schema formula_1.
        # "table_names" and "table_names_original" are not collated:
        #     "table_names": [ "races", "drivers", "status", "seasons", "constructors",
        #       "constructor standings", "results", "driver standings", "constructor results", "qualifying",
        #       "circuits", "pitstops", "laptimes" ],
        #     "table_names_original": [ "circuits", "races", "drivers", "status", "seasons", "constructors",
        #       "constructorStandings", "results", "driverStandings", "constructorResults", "qualifying",
        #       "pitStops", "lapTimes" ]
        # in that case, nl_name will be messed up for the table
        tables = [
            Table(name=orig, columns=columns_by_table[t_id], nl_name=name)
            for t_id, (orig, name) in enumerate(
                zip(schema_json["table_names_original"], schema_json["table_names"])
            )
        ]
        foreign_keys = [
            ForeignKey(col, other) for col, other in schema_json["foreign_keys"]
        ]
        db_id = schema_json["db_id"]
        values = []
        # Some tables mentioned in tables,json are not present in the download
        # This check catches errors when trying to read them.
        if os.path.exists(db_path + "/" + db_id):
            db_path = db_path + "/" + db_id + "/" + db_id + ".sqlite"
            for table in tables:
                for column in table.columns:
                    picklist = get_column_picklist(table.name, column.name, db_path)
                    values.extend(
                        [
                            val
                            for val in picklist
                            # this condition removes times from the list
                            if isinstance(val, str) and not val.count(":") == 2
                        ]
                    )

        return DbSchema(
            name=schema_json["db_id"],
            columns=columns,
            foreign_keys=foreign_keys,
            primary_keys=schema_json["primary_keys"],
            tables=tables,
            values=values,
        )


def load_schemas(
    schemas_path: StrPath = SCHEMAS_FILE, db_path: StrPath = COSQL_DIR / "database"
) -> Dict[str, DbSchema]:
    db_schema_details_file = str(db_path) + "/" + "db_schema_details.json"
    if os.path.exists(db_schema_details_file):
        with open(db_schema_details_file, "r") as db_schema_details_fp:
            return jsons.loads(db_schema_details_fp.read(), cls=Dict[str, DbSchema])

    with open(schemas_path) as tables_file:
        schemas_json = load(tables_file)
    schemas = [
        DbSchema.from_json(schema_json, str(db_path)) for schema_json in schemas_json
    ]
    return {schema.name: schema for schema in schemas}


if __name__ == "__main__":
    cosql_schemas = load_schemas(
        "/Users/subhrroy/workspace/pyharbor/data/benchclamp/raw/CoSQL/tables.json",
        "/Users/subhrroy/workspace/pyharbor/data/benchclamp/raw/CoSQL/database/",
    )
    with open(
        "/Users/subhrroy/workspace/pyharbor/data/benchclamp/raw/CoSQL/database/db_schema_details.json",
        "w",
    ) as fp:
        fp.write(jsons.dumps(cosql_schemas, cls=Dict[str, DbSchema]))

    spider_schemas = load_schemas(
        "/Users/subhrroy/workspace/pyharbor/data/benchclamp/raw/Spider/tables.json",
        "/Users/subhrroy/workspace/pyharbor/data/benchclamp/raw/Spider/database/",
    )
    with open(
        "/Users/subhrroy/workspace/pyharbor/data/benchclamp/raw/Spider/database/db_schema_details.json",
        "w",
    ) as fp:
        fp.write(jsons.dumps(cosql_schemas, cls=Dict[str, DbSchema]))
