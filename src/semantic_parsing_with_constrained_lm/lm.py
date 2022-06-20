# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import os
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Generic, List, Optional, Sequence, Tuple, TypeVar

import jsons
import torch

from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer

HS = TypeVar("HS")


# ClientType and TRAINED_MODEL_DIR are used in configs and in run_exp.py.
# TODO: Find a better long-term solution for this.
class ClientType(str, Enum):
    GPT3 = "GPT3"
    # finetuned GPT2
    GPT2 = "GPT2"
    # non-finetuned GPT2:
    GPT2Base = "GPT2Base"
    Mock = "Mock"
    BART = "Bart"
    # Use sm-gpt3-api for GPT-3
    SMGPT3 = "SMGPT3"


TRAINED_MODEL_DIR = os.environ.get("TRAINED_MODEL_DIR", "model/")


class AutoregressiveModel(Generic[HS], ABC):
    """Base class for language models and sequence-to-sequence models."""

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def tokenizer(self) -> ClampTokenizer:
        pass

    @abc.abstractmethod
    async def extend(
        self,
        tokens: Sequence[int],
        hidden_state: HS,
        drop_next_hidden_state: bool = False,
    ) -> Tuple[torch.Tensor, HS]:
        pass

    @abc.abstractmethod
    async def next_logprobs(self, hidden_state: HS) -> torch.Tensor:
        """Returns the distribution over the next token given the tokens in the hidden state."""
        pass

    async def __aenter__(self):
        pass

    async def __aexit__(self, *args):
        pass


class IncrementalLanguageModel(AutoregressiveModel[HS], ABC):
    @abc.abstractmethod
    async def execute(
        self,
        tokens: Sequence[int],
        hidden_state: Optional[HS] = None,
        drop_next_hidden_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[HS]]:
        """Run the language model on an input and a hidden state.

        tokens: List of token IDs, between 0 and `vocab_size - 1`.
        hidden_state: Returned from a previous call to this function.
        free_prev_hidden_state: Tell the backend that `hidden_state` is no longer needed after this call.
        drop_next_hidden_state: Tell the backend to not produce a return hidden state.

        Returns:
            a float32 tensor of size [seq len, vocab size] containing log probabilities.
            optionally, a hidden state that can be used in a future call to this function.
        """
        pass

    async def extend(
        self,
        tokens: Sequence[int],
        hidden_state: HS,
        drop_next_hidden_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[HS]]:
        return await self.execute(tokens, hidden_state, drop_next_hidden_state)

    async def logprob_of_completion(
        self, prefix_tokens: Sequence[int], completion_tokens: Sequence[int]
    ) -> float:
        """Return the log probability of `completion_tokens` following `prefix_tokens`."""
        logprobs, _ = await self.execute(
            tuple(prefix_tokens) + tuple(completion_tokens), drop_next_hidden_state=True
        )
        total_num_tokens = len(prefix_tokens) + len(completion_tokens)
        assert logprobs.shape[0] == len(prefix_tokens) + len(completion_tokens)

        score = (
            logprobs[
                range(len(prefix_tokens) - 1, total_num_tokens - 1), completion_tokens
            ]
            .sum()
            .item()
        )
        return score


class Seq2SeqModel(AutoregressiveModel[HS], ABC):
    @property
    @abc.abstractmethod
    def decoder_bos_ids(self) -> List[int]:
        """Return the IDs for tokens that should appear at the start of the decoder sequence."""
        pass

    @property
    @abc.abstractmethod
    def decoder_eos_id(self) -> int:
        """Return the ID for the token which signals that decoding is finished."""
        pass

    @abc.abstractmethod
    def encode_for_encoder(self, s: str) -> List[int]:
        """Convert string into a sequence of token IDs for input into the encoder."""
        pass

    @abc.abstractmethod
    def encode_prefix_for_decoder(
        self, s: str, include_bos_ids: bool = True
    ) -> List[int]:
        """Convert string into a sequence of token IDs for input into the decoder."""
        pass

    @abc.abstractmethod
    def decode_output(self, ids: Sequence[int]) -> str:
        """Convert token IDs from the decoder into a properly formatted string.

        The list of IDs should not contain the EOS token nor `decoder_bos_ids`."""
        pass

    @abc.abstractmethod
    async def initial(
        self,
        encoder_tokens: Sequence[int],
        decoder_tokens: Sequence[int],
        drop_next_hidden_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[HS]]:
        """Run the language model on tokens for the encoder and the decoder.

        encoder_tokens: List of token IDs, between 0 and `vocab_size - 1`.
        decoder_tokens: List of token IDs, between 0 and `vocab_size - 1`.
        drop_next_hidden_state: Tell the backend to not produce a return hidden state.

        Returns:
            a float32 tensor of size [decoder_tokens len, vocab size] containing log probabilities.
            optionally, a hidden state that can be used in a future call to this function.
        """
        pass


@dataclass
class Surround:
    bos: List[int]
    eos: List[int]
    starts_with_space: bool


@dataclass
class Seq2SeqSettings:
    input_surround: Surround
    output_surround: Surround
    decoder_start_token_id: Optional[int]


@dataclass
class Seq2SeqHelper:
    settings: Seq2SeqSettings
    tokenizer: ClampTokenizer
    decoder_start_token_ids: List[int]
    decoder_eos_token_id: int
    decoder_output_begins_with_space: bool

    @classmethod
    def from_settings_json(
        cls, json_str: str, tokenizer: ClampTokenizer
    ) -> "Seq2SeqHelper":
        settings = jsons.loads(json_str, cls=Seq2SeqSettings)
        decoder_start_token_ids = settings.output_surround.bos
        if settings.decoder_start_token_id is not None:
            decoder_start_token_ids = [
                settings.decoder_start_token_id
            ] + decoder_start_token_ids

        # Output surround eos should be one token id, since that needs to be used to detect end of output sequence.
        assert len(settings.output_surround.eos) == 1
        decoder_eos_token_id = settings.output_surround.eos[0]
        decoder_output_begins_with_space = settings.output_surround.starts_with_space
        return cls(
            settings,
            tokenizer,
            decoder_start_token_ids,
            decoder_eos_token_id,
            decoder_output_begins_with_space,
        )

    def encode_for_encoder(self, s: str) -> List[int]:
        string_to_tokenize = s
        if self.settings.input_surround.starts_with_space:
            string_to_tokenize = " " + s
        token_ids = (
            self.settings.input_surround.bos
            + list(self.tokenizer.encode(string_to_tokenize))
            + self.settings.input_surround.eos
        )
        return token_ids

    def encode_prefix_for_decoder(
        self, s: str, include_bos_ids: bool = True
    ) -> List[int]:
        string_to_tokenize = s
        if self.settings.output_surround.starts_with_space:
            string_to_tokenize = " " + s
        token_ids = self.tokenizer.encode(string_to_tokenize)
        if include_bos_ids:
            return self.decoder_start_token_ids + token_ids
        else:
            return token_ids

    def decode_output(self, ids: Sequence[int]) -> str:
        output = self.tokenizer.decode(list(ids))
        if self.decoder_output_begins_with_space and output[0] == " ":
            output = output[1:]
        return output


@dataclass
class TokensWithLogprobs:
    token_ids: torch.Tensor
    logprobs: torch.Tensor
