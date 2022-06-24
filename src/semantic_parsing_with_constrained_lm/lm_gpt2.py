# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, cast

import torch
import torch.nn.functional as F
from cached_property import cached_property
from transformers import PreTrainedModel

from semantic_parsing_with_constrained_lm.async_tools.batch_helper import BatchingHelper, BatchMaker
from semantic_parsing_with_constrained_lm.lm import (
    IncrementalLanguageModel,
    Seq2SeqHelper,
    Seq2SeqModel,
)
from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer


@dataclass
class GPT2State:
    prev_tokens: Tuple[int, ...]

    # Tuple of tuple(torch.FloatTensor) of length config.n_layers,
    # with each tuple having 2 tensors of shape
    #   (num_heads, sequence_length, embed_size_per_head)
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...] = dataclasses.field(
        repr=False
    )

    last_logprobs: torch.Tensor = dataclasses.field(repr=False)


@dataclass(eq=True, frozen=True)
class GPT2BatchMaker(BatchMaker):
    hidden_length: int
    extension_length: int

    model: PreTrainedModel = dataclasses.field(compare=False)

    @property
    def max_batch_size(self) -> int:
        # TODO: Revisit this entirely arbitrary number
        return 10

    @property
    def timeout(self) -> float:
        return 0.001

    @classmethod
    def from_args(
        cls, model: PreTrainedModel, args: Tuple[Sequence[int], Optional[GPT2State]]
    ):
        tokens, hidden_state = args
        if hidden_state is None:
            return cls(0, len(tokens), model=model)
        else:
            return cls(len(hidden_state.prev_tokens), len(tokens), model=model)

    async def execute(
        self, args: List[Tuple[Sequence[int], Optional[GPT2State]]]
    ) -> Tuple[torch.Tensor, List[GPT2State]]:
        if self.hidden_length > 0:
            return await self._execute_with_hidden_state(args)
        else:
            return await self._execute_without_hidden_state(args)

    async def _execute_without_hidden_state(
        self, args: List[Tuple[Sequence[int], Optional[GPT2State]]]
    ) -> Tuple[torch.Tensor, List[GPT2State]]:
        tokens, _ = cast(Tuple[Sequence[Sequence[int]], Any], zip(*args))
        input_ids = torch.tensor(
            list(tokens), dtype=torch.long, device=self.model.device
        )

        model_outputs = self.model(input_ids)
        logprobs = F.log_softmax(model_outputs["logits"], dim=-1)

        next_hidden_states = [
            GPT2State(
                tuple(tokens[i]),
                tuple(
                    cast(
                        Tuple[torch.Tensor, torch.Tensor],
                        tuple(
                            past_key_values_internal[i]
                            for past_key_values_internal in past_key_values_for_layer
                        ),
                    )
                    for past_key_values_for_layer in model_outputs["past_key_values"]
                ),
                logprobs[i, -1],
            )
            for i in range(len(args))
        ]
        return logprobs, next_hidden_states

    async def _execute_with_hidden_state(
        self, args: List[Tuple[Sequence[int], Optional[GPT2State]]]
    ):
        tokens, hidden_states = cast(
            Tuple[Sequence[Sequence[int]], Sequence[Optional[GPT2State]]], zip(*args)
        )
        assert not any(hs is None for hs in hidden_states)
        hidden_states = cast(Sequence[GPT2State], hidden_states)

        input_ids = torch.tensor(
            list(tokens), dtype=torch.long, device=self.model.device
        )
        past_key_values = tuple(
            tuple(
                torch.stack(
                    [
                        hidden_state.past_key_values[layer_i][kv_i]
                        for hidden_state in hidden_states
                    ]
                )
                for kv_i in range(2)
            )
            for layer_i in range(len(hidden_states[0].past_key_values))
        )

        model_outputs = self.model(input_ids=input_ids, past_key_values=past_key_values)
        logprobs = F.log_softmax(model_outputs["logits"], dim=-1)

        next_hidden_states = [
            GPT2State(
                past_hidden_state.prev_tokens + tuple(tokens[i]),
                tuple(
                    cast(
                        Tuple[torch.Tensor, torch.Tensor],
                        tuple(
                            past_key_values_internal[i]
                            for past_key_values_internal in past_key_values_for_layer
                        ),
                    )
                    for past_key_values_for_layer in model_outputs["past_key_values"]
                ),
                logprobs[i, -1],
            )
            for i, past_hidden_state in enumerate(hidden_states)
        ]
        return logprobs, next_hidden_states


@dataclass
class IncrementalGPT2(IncrementalLanguageModel[GPT2State]):
    pretrained_model_dir: str
    model: PreTrainedModel
    clamp_tokenizer: ClampTokenizer

    batch_helper: BatchingHelper[
        Tuple[Sequence[int], Optional[GPT2State]], Tuple[torch.Tensor, List[GPT2State]]
    ] = dataclasses.field(init=False)

    def __post_init__(self):
        self.batch_helper = BatchingHelper(
            lambda args: GPT2BatchMaker.from_args(self.model, args),
        )

    @cached_property
    def vocab_size(self) -> int:  # pylint: disable=invalid-overridden-method
        return self.clamp_tokenizer.vocab_size

    @cached_property
    def tokenizer(self) -> ClampTokenizer:  # pylint: disable=invalid-overridden-method
        return self.clamp_tokenizer

    async def execute(
        self,
        tokens: Sequence[int],
        hidden_state: Optional[GPT2State] = None,
        drop_next_hidden_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[GPT2State]]:
        (batched_logprobs, next_hidden_states), i = await self.batch_helper.execute(
            (tokens, hidden_state)
        )
        return (
            batched_logprobs[i, : len(tokens)],
            None if drop_next_hidden_state else next_hidden_states[i],
        )

    async def next_logprobs(self, hidden_state: GPT2State) -> torch.Tensor:
        return hidden_state.last_logprobs


@dataclass
class Seq2SeqGPT2(Seq2SeqModel[GPT2State]):
    """Useful for using GPT-2 as a fine-tuned seq2seq model."""

    pretrained_model_dir: str
    model: PreTrainedModel
    clamp_tokenizer: ClampTokenizer
    incremental_model: IncrementalGPT2 = dataclasses.field(init=False)
    seq2seq_helper: Seq2SeqHelper = dataclasses.field(init=False)

    def __post_init__(self):
        self.incremental_model = IncrementalGPT2(
            pretrained_model_dir=self.pretrained_model_dir,
            model=self.model,
            clamp_tokenizer=self.clamp_tokenizer,
        )
        with open(
            os.path.join(self.pretrained_model_dir, "seq2seq_settings.json")
        ) as settings_f:
            self.seq2seq_helper = Seq2SeqHelper.from_settings_json(
                settings_f.read(), self.incremental_model.tokenizer
            )

    @cached_property
    def vocab_size(self) -> int:  # pylint: disable=invalid-overridden-method
        return self.incremental_model.vocab_size

    @cached_property
    def tokenizer(self) -> ClampTokenizer:  # pylint: disable=invalid-overridden-method
        return self.clamp_tokenizer

    @property
    def decoder_bos_ids(self) -> List[int]:
        return self.seq2seq_helper.decoder_start_token_ids

    @property
    def decoder_eos_id(self) -> int:
        return self.seq2seq_helper.decoder_eos_token_id

    def encode_for_encoder(self, s: str) -> List[int]:
        return self.seq2seq_helper.encode_for_encoder(s)

    def encode_prefix_for_decoder(
        self, s: str, include_bos_ids: bool = True
    ) -> List[int]:
        return self.seq2seq_helper.encode_prefix_for_decoder(s, include_bos_ids)

    def decode_output(self, ids: Sequence[int]) -> str:
        return self.seq2seq_helper.decode_output(ids)

    async def initial(
        self,
        encoder_tokens: Sequence[int],
        decoder_tokens: Sequence[int],
        drop_next_hidden_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[GPT2State]]:
        logprobs, hidden_state = await self.incremental_model.execute(
            tuple(encoder_tokens) + tuple(decoder_tokens), None, drop_next_hidden_state
        )
        return logprobs[len(encoder_tokens) :], hidden_state

    async def extend(
        self,
        tokens: Sequence[int],
        hidden_state: GPT2State,
        drop_next_hidden_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[GPT2State]]:
        return await self.incremental_model.execute(
            tokens, hidden_state, drop_next_hidden_state
        )

    async def next_logprobs(self, hidden_state: GPT2State) -> torch.Tensor:
        return hidden_state.last_logprobs
