# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import (
    BartForConditionalGeneration,
    GPT2LMHeadModel,
    PreTrainedModel,
    T5ForConditionalGeneration,
)

from semantic_parsing_with_constrained_lm.lm import Seq2SeqSettings, Surround
from semantic_parsing_with_constrained_lm.tokenization import (
    ClampTokenizer,
    GPT2ClampTokenizer,
    T5ClampTokenizer,
)


class TrainedModelNotFoundError(FileNotFoundError):
    pass


@dataclass  # type: ignore
class ClampModelConfig(abc.ABC):
    model_id: str
    model_loc: Path
    device_map: Optional[Dict[int, List[int]]] = None

    @abstractmethod
    def setup_model(self) -> Tuple[PreTrainedModel, ClampTokenizer, Seq2SeqSettings]:
        pass

    def maybe_parallelize(self, model: PreTrainedModel) -> None:
        if torch.cuda.is_available():
            if self.device_map is not None:
                print(f"Parallelizing model with {self.device_map}")
                model.parallelize(self.device_map)
            else:
                print("Entire model to GPU 0")
                model.to(torch.device("cuda:0"))
        else:
            model.to(torch.device("cpu"))


class BartModelConfig(ClampModelConfig):
    def setup_model(self) -> Tuple[PreTrainedModel, ClampTokenizer, Seq2SeqSettings]:
        if not self.model_loc.exists():
            raise TrainedModelNotFoundError(
                f"Model files not found in {self.model_loc}"
            )
        model = BartForConditionalGeneration.from_pretrained(self.model_loc)
        tokenizer = GPT2ClampTokenizer.from_pretrained(str(self.model_loc))
        seq2seq_settings = Seq2SeqSettings(
            input_surround=Surround(bos=[0], eos=[2], starts_with_space=True),
            output_surround=Surround(bos=[0], eos=[2], starts_with_space=True),
            decoder_start_token_id=2,
        )
        self.maybe_parallelize(model)
        model.eval()
        return model, tokenizer, seq2seq_settings


class T5ModelConfig(ClampModelConfig):
    def setup_model(self) -> Tuple[PreTrainedModel, ClampTokenizer, Seq2SeqSettings]:
        if not self.model_loc.exists():
            raise TrainedModelNotFoundError(
                f"Model files not found in {self.model_loc}"
            )
        print(f"Loading model from {self.model_loc}")
        model = T5ForConditionalGeneration.from_pretrained(self.model_loc)
        print("Done")
        tokenizer = T5ClampTokenizer.from_pretrained(str(self.model_loc))
        seq2seq_settings = Seq2SeqSettings(
            input_surround=Surround(bos=[], eos=[1], starts_with_space=True),
            output_surround=Surround(bos=[], eos=[1], starts_with_space=True),
            decoder_start_token_id=tokenizer.pad_token_id,
        )
        self.maybe_parallelize(model)
        model.eval()
        return model, tokenizer, seq2seq_settings


class CodeT5ModelConfig(ClampModelConfig):
    def setup_model(self) -> Tuple[PreTrainedModel, ClampTokenizer, Seq2SeqSettings]:
        if not self.model_loc.exists():
            raise TrainedModelNotFoundError(
                f"Model files not found in {self.model_loc}"
            )
        model = T5ForConditionalGeneration.from_pretrained(self.model_loc)
        tokenizer = GPT2ClampTokenizer.from_pretrained(str(self.model_loc))
        seq2seq_settings = Seq2SeqSettings(
            input_surround=Surround(bos=[1], eos=[2], starts_with_space=True),
            output_surround=Surround(bos=[1], eos=[2], starts_with_space=True),
            decoder_start_token_id=0,
        )
        self.maybe_parallelize(model)
        model.eval()
        return model, tokenizer, seq2seq_settings


class GPT2ModelConfig(ClampModelConfig):
    def setup_model(self) -> Tuple[PreTrainedModel, ClampTokenizer, Seq2SeqSettings]:
        if not self.model_loc.exists():
            raise TrainedModelNotFoundError(
                f"Model files not found in {self.model_loc}"
            )
        model = GPT2LMHeadModel.from_pretrained(self.model_loc)
        tokenizer = GPT2ClampTokenizer.from_pretrained(str(self.model_loc))
        seq2seq_settings = Seq2SeqSettings(
            input_surround=Surround(
                bos=[20490, 25], eos=[198], starts_with_space=True
            ),  # bos: "Human:", eos: "\n"
            output_surround=Surround(
                bos=[34556, 25], eos=[198], starts_with_space=True
            ),  # bos: "Computer:", eos: "\n"
            decoder_start_token_id=None,
        )
        self.maybe_parallelize(model)
        model.eval()
        return model, tokenizer, seq2seq_settings
