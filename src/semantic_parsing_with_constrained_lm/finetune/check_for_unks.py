# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib
from typing import Callable, Dict, List, Optional, Set

import typer

from semantic_parsing_with_constrained_lm.lm import Surround
from semantic_parsing_with_constrained_lm.run_exp import filter_exp_dict
from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer
from semantic_parsing_with_constrained_lm.finetune.lm_finetune import TrainExperiment


def find_and_record_unk_tokens(
    tokenizer: ClampTokenizer,
    surround: Surround,
    ids: List[int],
    orig: str,
    unk_tokens: Set[bytes],
) -> None:
    if surround.starts_with_space:
        orig = " " + orig
    tokens = tokenizer.tokenize(orig)
    ids = ids[len(surround.bos) : len(ids) - len(surround.eos)]
    assert len(tokens) == len(ids)

    for token_bytes, token_id in zip(tokens, ids):
        if token_id == tokenizer.unk_token_id:
            unk_tokens.add(token_bytes)


def main(
    config_name: str = typer.Option(...),
    exp_names: Optional[List[str]] = typer.Option(None),
    exp_name_pattern: Optional[List[str]] = typer.Option(None),
):
    config_mod = importlib.import_module(config_name)
    exps: Dict[str, Callable[[], TrainExperiment]] = config_mod.build_config()  # type: ignore
    filtered_exp_dict = filter_exp_dict(exps, exp_names, exp_name_pattern)

    for exp_name in filtered_exp_dict:
        exp = filtered_exp_dict[exp_name]()
        unk_id = exp.tokenizer.unk_token_id
        assert unk_id is not None

        train_dataset = exp.make_clamp_dataset(exp.train_data)
        num_unks_in_inputs = 0
        num_unks_in_labels = 0

        unk_tokens: Set[bytes] = set()

        for i, _ in enumerate(train_dataset):  # type: ignore
            datum = train_dataset[i]
            input_ids = datum["input_ids"]
            labels = datum["labels"]

            if any(t == unk_id for t in input_ids):
                num_unks_in_inputs += 1
                find_and_record_unk_tokens(
                    exp.tokenizer,
                    exp.seq2seq_settings.input_surround,
                    input_ids,
                    exp.train_data[i].natural,
                    unk_tokens,
                )

            if any(t == unk_id for t in labels):
                num_unks_in_labels += 1
                find_and_record_unk_tokens(
                    exp.tokenizer,
                    exp.seq2seq_settings.output_surround,
                    labels,
                    exp.train_data[i].canonical,
                    unk_tokens,
                )

        print(
            f"{exp_name}: {num_unks_in_inputs}/{len(train_dataset)} unks in inputs, "
            f"{num_unks_in_labels}/{len(train_dataset)} unks in labels."
        )
        print(f"unk_tokens = {unk_tokens}")


if __name__ == "__main__":
    typer.run(main)
