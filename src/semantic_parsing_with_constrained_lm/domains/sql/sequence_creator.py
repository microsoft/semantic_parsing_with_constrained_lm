# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing_extensions import Literal

from semantic_parsing_with_constrained_lm.datum import BenchClampDatum
from semantic_parsing_with_constrained_lm.sequence_creator import SequenceCreator
from semantic_parsing_with_constrained_lm.tokenization import GPT2ClampTokenizer


class CoSqlUtterance(SequenceCreator):
    def __init__(
        self, use_db_val: bool, past_utterances: Literal["none", "one", "all"]
    ):
        self.use_db_val = use_db_val
        self.past_utterances = past_utterances
        self.gpt2_tokenizer = GPT2ClampTokenizer.from_pretrained("gpt2")

    def create_sequence(self, datum: BenchClampDatum) -> str:
        db_schema = (
            datum.db_schema_with_val if self.use_db_val else datum.db_schema_without_val
        )
        all_past_utterances = datum.utterance.split(" | ")
        current_utterance = all_past_utterances[-1]
        if self.past_utterances == "none" or len(all_past_utterances) == 1:
            past_utterances = ""
        elif self.past_utterances == "one":
            past_utterances = all_past_utterances[-2]
        else:
            past_utterances = " | ".join(all_past_utterances[:-1])

        sequence = " , ".join([past_utterances, db_schema, current_utterance])  # type: ignore
        sequence_token_ids = self.gpt2_tokenizer.encode(sequence)
        # start_index = max(0, len(sequence_token_ids) - 1000)
        start_index = 0
        return self.gpt2_tokenizer.decode(sequence_token_ids[start_index:])
