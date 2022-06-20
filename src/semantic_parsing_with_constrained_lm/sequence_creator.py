# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod

from semantic_parsing_with_constrained_lm.datum import BenchClampDatum


class SequenceCreator(ABC):
    @abstractmethod
    def create_sequence(self, datum: BenchClampDatum) -> str:
        pass


class IdentitySequenceCreator(SequenceCreator):
    def create_sequence(self, datum: BenchClampDatum) -> str:
        return datum.utterance
