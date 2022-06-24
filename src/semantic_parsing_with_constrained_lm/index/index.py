# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import Generic, Iterable, Tuple, TypeVar

Key = TypeVar("Key")
Query = TypeVar("Query")
Candidate = TypeVar("Candidate")


class Index(Generic[Key, Query], ABC):
    """
    Encapsulates any index that can be searched over.
    It can either be a sparse index (e.g. powered by Whoosh), or a dense index (e.g. powered by FAISS).
    """

    @abstractmethod
    def search(self, query: Query, top_k: int) -> Iterable[Tuple[Key, float]]:
        raise NotImplementedError


class DynamicIndex(Generic[Key, Query, Candidate], Index[Key, Query]):
    """
    Any index that supports dynamic addition to the set of candidates.
    """

    @abstractmethod
    def add(self, candidates: Iterable[Candidate]) -> None:
        raise NotImplementedError
