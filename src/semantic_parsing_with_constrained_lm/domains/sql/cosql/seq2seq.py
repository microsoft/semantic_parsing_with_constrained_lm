# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
ported from
https://github.com/ElementAI/picard/blob/5ddd6cb9f74efca87d4604d5ddddc1b638459466/seq2seq/utils/cosql.py#L10
Under the Apache 2 licence:
https://github.com/ElementAI/picard/blob/main/LICENSE
"""
from typing import Any, Dict, List

from semantic_parsing_with_constrained_lm.domains.sql.cosql.content_encoder import (
    get_database_matches,
)


def get_input(
    utterances: List[str], serialized_schema: str, prefix: str, sep: str = " | "
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
        prefix
        + utterances[-1].strip()
        + " "
        + serialized_schema.strip()
        + serialized_reversed_utterance_head
    )


def serialize_schema(
    question: str,
    db_path: str,
    db_id: str,
    db_column_names: Dict[str, Any],
    db_table_names: List[str],
    schema_serialization_with_db_content: bool = True,
) -> str:
    # schema_serialization_with_db_id: bool = True
    # schema_serialization_randomized: bool = False
    # normalize_query: bool = True
    # schema_serialization_type = "peteshaw"
    # see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/append_schema.py#L42
    db_id_str = " | {db_id}"
    table_sep = ""
    table_str = " | {table} : {columns}"
    column_sep = " , "
    column_str_with_values = "{column} ( {values} )"
    column_str_without_values = "{column}"
    value_sep = " , "

    def get_column_str(table_name: str, column_name: str) -> str:
        column_name_str = column_name.lower()
        if schema_serialization_with_db_content:
            matches = get_database_matches(
                question=question,
                table_name=table_name,
                column_name=column_name,
                db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
            )
            if matches:
                return column_str_with_values.format(
                    column=column_name_str, values=value_sep.join(matches)
                )
            else:
                return column_str_without_values.format(column=column_name_str)
        else:
            return column_str_without_values.format(column=column_name_str)

    tables = [
        table_str.format(
            table=table_name.lower(),
            columns=column_sep.join(
                map(
                    lambda y: get_column_str(
                        table_name=table_name,  # pylint: disable=cell-var-from-loop
                        column_name=y[1],
                    ),
                    filter(
                        lambda y: y[0]
                        == table_id,  # pylint: disable=cell-var-from-loop
                        zip(
                            db_column_names["table_id"], db_column_names["column_name"]
                        ),
                    ),
                )
            ),
        )
        for table_id, table_name in enumerate(db_table_names)
    ]
    return db_id_str.format(db_id=db_id) + table_sep.join(tables)
