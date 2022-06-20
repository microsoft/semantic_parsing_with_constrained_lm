# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from semantic_parsing_with_constrained_lm.datum import BenchClampDatum
from semantic_parsing_with_constrained_lm.sequence_creator import SequenceCreator


class LastAgentUtterance(SequenceCreator):
    def create_sequence(self, datum: BenchClampDatum) -> str:
        last_agent_utterance = (
            datum.last_agent_utterance if datum.last_agent_utterance is not None else ""
        )
        return " | ".join([last_agent_utterance, datum.utterance])


class LastUserAgentUtterance(SequenceCreator):
    def create_sequence(self, datum: BenchClampDatum) -> str:
        last_agent_utterance = (
            datum.last_agent_utterance if datum.last_agent_utterance is not None else ""
        )
        last_user_utterance = (
            datum.last_user_utterance if datum.last_user_utterance is not None else ""
        )
        return " | ".join([last_user_utterance, last_agent_utterance, datum.utterance])
