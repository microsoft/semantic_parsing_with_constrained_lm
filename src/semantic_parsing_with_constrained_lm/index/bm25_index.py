# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import tempfile
from typing import Callable, Generic, Iterable, List, Sequence, Tuple

import whoosh.index
from whoosh.fields import STORED, TEXT, SchemaClass
from whoosh.qparser import OrGroup, QueryParser

from semantic_parsing_with_constrained_lm.datum import DatumSub, FullDatumSub
from semantic_parsing_with_constrained_lm.index import Candidate, DynamicIndex, Query
from semantic_parsing_with_constrained_lm.model import DataRetriever


class PromptSchema(SchemaClass):
    text = TEXT()
    key = STORED()


class BM25Index(Generic[Query, Candidate], DynamicIndex[int, Query, Candidate]):
    def __init__(
        self, get_content: Callable[[Candidate], str], get_query: Callable[[Query], str]
    ):
        # TODO: custom tokenizer from ClampTokenizer
        # TODO: now indexed in a temp dir
        tmp_indexer_loc = tempfile.mkdtemp()
        self.index = whoosh.index.create_in(tmp_indexer_loc, schema=PromptSchema)
        self.get_content = get_content
        self.get_query = get_query

    @classmethod
    def create(
        cls,
        candidates: Iterable[Candidate],
        get_content: Callable[[Candidate], str],
        get_query: Callable[[Query], str],
    ) -> "BM25Index":
        index = BM25Index(get_content, get_query)
        with index.index.writer() as writer:
            for i, candidate in enumerate(candidates):
                writer.add_document(text=get_content(candidate), key=i)
        return index

    def add(self, candidates: Iterable[Candidate]):
        n = self.index.doc_count()
        with self.index.writer() as writer:
            for i, candidate in enumerate(candidates):
                writer.add_document(
                    text=self.get_content(candidate), key=n + i
                )  # auto-increment key

    def search(self, query: Query, top_k: int = 10) -> List[Tuple[int, float]]:
        with self.index.searcher() as searcher:
            query_parser = QueryParser("text", schema=searcher.schema, group=OrGroup)
            q = query_parser.parse(self.get_query(query))
            results = searcher.search(q, limit=top_k)
            return [(result["key"], result.score) for result in results]


class BM25Retriever(DataRetriever[FullDatumSub, DatumSub]):
    def __init__(
        self,
        train_data: Sequence[FullDatumSub],
        top_k: int = 20,
        best_first: bool = True,
        seed: int = 12345,
    ):
        self.index: BM25Index[DatumSub, FullDatumSub] = BM25Index.create(
            train_data,
            get_content=lambda c: c.natural,  # type: ignore
            get_query=lambda q: q.natural,  # type: ignore
        )
        self.data: List[FullDatumSub] = list(train_data)
        self.top_k = top_k
        self.best_first = best_first
        self.prng = random.Random(
            seed
        )  # a random number generator to ensure deterministic behavior

    def augment_with_random_samples(
        self, data: Sequence[FullDatumSub], retrieved_keys: Sequence[int]
    ) -> Sequence[FullDatumSub]:

        if len(retrieved_keys) < self.top_k:
            print(
                f"Could not retrieve {self.top_k} examples, got only {len(retrieved_keys)}"
            )
            keys_to_sample = sorted(set(range(len(data))).difference(retrieved_keys))
            sampled_keys = self.prng.sample(
                keys_to_sample,
                k=min(self.top_k - len(retrieved_keys), len(keys_to_sample)),
            )
            augmented_keys = list(retrieved_keys) + list(sampled_keys)
            print(f"Added samples to make it of size {len(augmented_keys)}")
            items = [data[k] for k in augmented_keys[: self.top_k]]
        else:
            items = [data[k] for k in retrieved_keys[: self.top_k]]

        return items if self.best_first else list(reversed(items))

    def add(self, data: Sequence[FullDatumSub]):
        self.index.add(data)
        self.data.extend(data)

    async def __call__(self, test_datum: DatumSub) -> Sequence[FullDatumSub]:
        results = self.augment_with_random_samples(
            data=self.data,
            retrieved_keys=[
                key for key, _ in self.index.search(test_datum, top_k=self.top_k)
            ],  # score discarded
        )
        return results
