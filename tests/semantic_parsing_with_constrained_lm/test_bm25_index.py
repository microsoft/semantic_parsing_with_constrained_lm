# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
from pathlib import Path

import pytest

from semantic_parsing_with_constrained_lm.datum import Datum
from semantic_parsing_with_constrained_lm.domains.calflow import (
    CalflowOutputLanguage,
    read_calflow_jsonl,
)
from semantic_parsing_with_constrained_lm.index.bm25_index import BM25Retriever
from semantic_parsing_with_constrained_lm.paths import DOMAINS_DIR

TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXAMPLES_DIR = DOMAINS_DIR / "calflow/data"


class TestBM25Index:
    @pytest.fixture(scope="class")
    def index(self) -> BM25Retriever:
        return BM25Retriever(
            read_calflow_jsonl(
                TEST_DATA_DIR / "dev_top.jsonl", CalflowOutputLanguage.Lispress
            ),
            top_k=2,
        )

    def test_retrieve(self, index: BM25Retriever):
        query = Datum(
            natural="Who is going to dinner tonight with me?",
            dialogue_id=None,
            turn_part_index=0,
            agent_context="",
        )
        retrieved = asyncio.run(index(query))
        docs = [candidate.natural for candidate in retrieved]
        assert docs == [
            "Who is coming to dinner with me tonight?",
            "Who is a maybe for dinner ?",
        ]


if __name__ == "__main__":
    t = TestBM25Index()
    t.test_retrieve(t.index())
