# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Finetunes language models for CLAMP task using transformers Trainer interface. """
import dataclasses
import glob
import importlib
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import jsons
import torch
import typer
from torch import tensor
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedModel, Trainer, TrainingArguments

import semantic_parsing_with_constrained_lm
from semantic_parsing_with_constrained_lm.util import logger
from semantic_parsing_with_constrained_lm.datum import FullDatum
from semantic_parsing_with_constrained_lm.lm import Seq2SeqSettings
from semantic_parsing_with_constrained_lm.run_exp import filter_exp_dict
from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer

PRETRAINED_MODEL_DIR = os.environ.get("PRETRAINED_MODEL_DIR", "")
MAX_OUTPUT_SEQUENCE_FOR_TRAINING = 500


@dataclass
class DataCollatorForSeq2Seq:
    """Simple Data collator that works with ClampDataset items."""

    pad_token_id: int = -100

    def __call__(self, features: List[Dict]):
        inputs_with_labels = []
        skips = []
        skips_present = False
        # Removing examples with output length > MAX_OUTPUT_SEQUENCE_FOR_TRAINING
        # Trimming from start for long input sequences
        for feature in features:
            if len(feature["labels"]) <= MAX_OUTPUT_SEQUENCE_FOR_TRAINING:
                # start_index = max(0, len(feature["input_ids"]) - MAX_INPUT_SEQUENCE_FOR_TRAINING)
                inputs_with_labels.append((feature["input_ids"], feature["labels"]))
            else:
                skips.append((len(feature["input_ids"]), len(feature["labels"])))
                skips_present = True

        if skips_present:
            print(f"Skipping: {skips}")

        labels_list = [labels for _, labels in inputs_with_labels]
        max_labels_length = max(len(labels) for labels in labels_list)
        input_ids_list = [inputs for inputs, _ in inputs_with_labels]
        max_input_length = max(len(input_ids) for input_ids in input_ids_list)
        attention_mask = []
        padded_input_ids = []
        padded_labels = []
        for input_ids in input_ids_list:
            padded_input_ids.append(
                input_ids + [self.pad_token_id] * (max_input_length - len(input_ids))
            )
            attention_mask.append(
                [1] * len(input_ids) + [0] * (max_input_length - len(input_ids))
            )
        for labels in labels_list:
            # -100 will remove the labels from loss computation
            padded_labels.append(labels + [-100] * (max_labels_length - len(labels)))
        return {
            "input_ids": tensor(padded_input_ids, dtype=torch.long),
            "labels": tensor(padded_labels, dtype=torch.long),
            "attention_mask": tensor(attention_mask, dtype=torch.long),
        }


@dataclass
class TrainExperiment:
    train_data: Sequence[FullDatum]
    eval_data: Sequence[FullDatum]
    model: PreTrainedModel
    tokenizer: ClampTokenizer
    seq2seq_settings: Seq2SeqSettings
    is_encoder_decoder: bool
    training_args: TrainingArguments
    log_dir: Path

    def make_clamp_dataset(self, data: Sequence[FullDatum]) -> "ClampDataset":
        return ClampDataset(
            data=data,
            tokenizer=self.tokenizer,
            is_encoder_decoder=self.is_encoder_decoder,
            seq2seq_settings=self.seq2seq_settings,
        )


@dataclass
class ClampDataset(Dataset):
    data: Sequence[FullDatum]
    tokenizer: ClampTokenizer
    is_encoder_decoder: bool
    seq2seq_settings: Seq2SeqSettings

    cache: Dict[int, Dict[str, Any]] = dataclasses.field(default_factory=dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cached = self.cache.get(idx)
        if cached is not None:
            return cached

        datum = self.data[idx]
        natural = (
            " " + datum.natural
            if self.seq2seq_settings.input_surround.starts_with_space
            else datum.natural
        )
        input_token_ids = (
            self.seq2seq_settings.input_surround.bos
            + self.tokenizer.encode(natural)
            + self.seq2seq_settings.input_surround.eos
        )
        canonical = (
            " " + datum.canonical
            if self.seq2seq_settings.output_surround.starts_with_space
            else datum.canonical
        )
        output_token_ids = (
            self.seq2seq_settings.output_surround.bos
            + self.tokenizer.encode(canonical)
            + self.seq2seq_settings.output_surround.eos
        )
        if self.is_encoder_decoder:
            result = {
                "input_ids": input_token_ids,
                "labels": output_token_ids,
                "length": len(input_token_ids),
            }
        else:
            result = {
                "input_ids": input_token_ids + output_token_ids,
                # -100 is ignored while computing loss
                "labels": [-100] * len(input_token_ids) + output_token_ids,
                "length": len(input_token_ids),
            }
        self.cache[idx] = result
        return result


def reset_gpu_stats(msg: str) -> None:
    print(msg)
    if torch.cuda.is_available():
        for gpu_id in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(gpu_id)


def print_gpu_stats(msg: str) -> None:
    print(msg)
    if torch.cuda.is_available():
        for gpu_id in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            max_allocated_memory = torch.cuda.max_memory_allocated(gpu_id)
            print(
                f"Max memory GPU {gpu_id}:",
                max_allocated_memory,
                "/",
                total_memory,
                "=",
                max_allocated_memory / total_memory,
            )


def run(
    exp_name: str, exp: TrainExperiment, log_dir: Optional[pathlib.Path] = None
) -> None:
    print(f"Running train experiment {exp_name} with train size {len(exp.train_data)}")
    print(f"Batch size per device: {exp.training_args.per_device_train_batch_size}")
    print(
        f"Gradient accumulation steps: {exp.training_args.gradient_accumulation_steps}"
    )
    reset_gpu_stats("")
    train_output_sequence_lengths = [
        len(exp.tokenizer.encode(" " + datum.canonical)) for datum in exp.train_data
    ]
    print(
        f"Number of examples removed on output truncation at length {MAX_OUTPUT_SEQUENCE_FOR_TRAINING}:",
        sum(
            [
                1 if length > MAX_OUTPUT_SEQUENCE_FOR_TRAINING else 0
                for length in train_output_sequence_lengths
            ]
        ),
    )
    train_dataset = exp.make_clamp_dataset(exp.train_data)
    eval_dataset = exp.make_clamp_dataset(exp.eval_data)
    if log_dir is not None:
        exp.training_args.logging_dir = log_dir / exp_name
        exp.log_dir = log_dir / exp_name

    print_gpu_stats("Before Training GPU State")
    trainer = Trainer(
        model=exp.model,
        args=exp.training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(exp.tokenizer.pad_token_id),
    )
    trainer.train()
    print_gpu_stats("After Training GPU State")

    # save tokenizer and seq2seq_settings in each checkpoint directory
    for checkpoint_dir in glob.glob(f"{exp.training_args.output_dir}/checkpoint-*/"):
        exp.tokenizer.save_pretrained(checkpoint_dir)
        with open(f"{checkpoint_dir}/seq2seq_settings.json", "w") as settings_f:
            settings_f.write(jsons.dumps(exp.seq2seq_settings))

    print(f"Train experiment {exp_name} completed.")
    del exp
    del train_dataset
    del eval_dataset
    del trainer
    torch.cuda.empty_cache()


def main(
    config_name: str = typer.Option(...),
    exp_names: Optional[List[str]] = typer.Option(None),
    exp_name_pattern: Optional[List[str]] = typer.Option(None),
    log_dir: Optional[pathlib.Path] = typer.Option(None),
    batch_size: Optional[int] = typer.Option(None),
):
    config_mod = importlib.import_module(config_name)
    exps = config_mod.build_config(log_dir)  # type: ignore
    filtered_exp_dict = filter_exp_dict(exps, exp_names, exp_name_pattern)
    for exp_name in filtered_exp_dict:
        now = datetime.now().strftime("%Y%m%dT%H%M%S")
        try:
            print(f"Instantiating {exp_name}")
            exp = filtered_exp_dict[exp_name]()
        except FileNotFoundError as err:
            # Trying to load models before training and saving them.
            print(err)
            continue

        if isinstance(exp, semantic_parsing_with_constrained_lm.finetune.lm_finetune.TrainExperiment):  # type: ignore
            exp_log_dir = exp.log_dir / exp_name
            exp_log_dir.mkdir(exist_ok=True, parents=True)
            if batch_size is not None:
                exp.training_args.per_device_train_batch_size = batch_size
                exp.training_args.gradient_accumulation_steps = 32 // batch_size
            with logger.intercept_output(
                Path(f"{exp_log_dir}/stdout.{now}"),
                Path(f"{exp_log_dir}/stderr.{now}"),
            ):
                run(exp_name, exp, log_dir)
        else:
            del exp
            torch.cuda.empty_cache()


if __name__ == "__main__":
    typer.run(main)
