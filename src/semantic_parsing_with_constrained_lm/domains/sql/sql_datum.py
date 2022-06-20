# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass

from semantic_parsing_with_constrained_lm.datum import FullDatum


@dataclass(frozen=True)
class SqlDatum(FullDatum):
    schema_name: str
