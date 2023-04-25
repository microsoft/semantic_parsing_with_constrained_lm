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

class OracleRewrittenUtterance(SequenceCreator):
    # The rewritten utterance written by the data specialists.
    def create_sequence(self, datum: BenchClampDatum) -> str:
        oracle_rewritten_utterance = (
            datum.oracle_rewritten_utterance if datum.oracle_rewritten_utterance is not None else datum.utterance
        )
        return oracle_rewritten_utterance

class RewrittenUtterance(SequenceCreator):
    # The rewritten utterance written predicted by model
    def create_sequence(self, datum: BenchClampDatum) -> str:
        rewritten_utterance = (
            datum.rewritten_utterance if datum.rewritten_utterance is not None else datum.utterance
        )
        return rewritten_utterance.replace("\n", "")

class LastPlan(SequenceCreator):
    def create_sequence(self, datum: BenchClampDatum) -> str:
        last_plan = (
           datum.last_plan if datum.last_plan is not None else ""
        )
        return " | ".join([last_plan, datum.utterance])
