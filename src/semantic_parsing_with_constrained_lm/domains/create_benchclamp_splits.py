# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from dataflow.core.io_utils import save_jsonl_file

from semantic_parsing_with_constrained_lm.datum import BenchClampDatum, Datum, FullDatum
from semantic_parsing_with_constrained_lm.decoding.partial_parse import PartialParse
from semantic_parsing_with_constrained_lm.tokenization import GPT2ClampTokenizer


def can_force_decode(
    clamp_tokenizer: GPT2ClampTokenizer,
    partial_parse_builder: Callable[[Datum], PartialParse],
    datum: FullDatum,
) -> bool:
    token_ids = clamp_tokenizer.encode(" " + datum.canonical)
    partial_parse = partial_parse_builder(datum)  # type: ignore
    for i, token_id in enumerate(token_ids):
        next_tokens, can_end = partial_parse.allowed_next(
            torch.tensor([token_id]), top_k=1
        )
        if token_id not in next_tokens:  # type: ignore
            print()
            print(f"Input:  {repr(datum.natural)}")
            print(f"Output: {repr(datum.canonical)}")
            print(f"Prefix: {repr(clamp_tokenizer.decode(token_ids[:i]))}")
            print(f"Rejected: {clamp_tokenizer.decode([token_id])}")
            return False
        partial_parse = partial_parse.append(token_id)
    next_tokens, can_end = partial_parse.allowed_next(torch.tensor([0]), top_k=1)
    if not can_end:
        print()
        print(f"Input:  {repr(datum.natural)}")
        print(f"Output: {repr(datum.canonical)}")
        print("Ending not allowed here")
        return False

    return True


def gather_subset(data: List[BenchClampDatum], size: int) -> List[BenchClampDatum]:
    dialogue_id_to_turn_count: Dict[str, int] = defaultdict(int)
    for datum in data:
        dialogue_id_to_turn_count[datum.dialogue_id] += 1  # type: ignore

    dialogue_ids = sorted(dialogue_id_to_turn_count.keys())
    random.shuffle(dialogue_ids)
    selected_dialogue_ids = []
    num_turns_covered = 0
    for dialogue_id in dialogue_ids:
        selected_dialogue_ids.append(dialogue_id)
        num_turns_covered += dialogue_id_to_turn_count[dialogue_id]
        if num_turns_covered >= size:
            break

    if num_turns_covered < size:
        print(f"Not enough data to create subset of size {size}")

    selected_dialogue_ids_set = set(selected_dialogue_ids)
    return [datum for datum in data if datum.dialogue_id in selected_dialogue_ids_set]


def create_benchclamp_splits(
    train_data: List[BenchClampDatum],
    dev_data: List[BenchClampDatum],
    test_data: Optional[List[BenchClampDatum]],
    output_dir: Path,
):
    """
    Sample splits for BenchClamp experiments.
    1. 5 low data train splits of size 500, single dev set of size 50
    2. 3 medium data train splits of size 5000, single dev set of size 500
    3. Full data split. Reuses dev set for medium data.
    4. Single test split of size 2000.
    """
    random.seed(0)
    if test_data is None:
        train_dialogue_ids = sorted({datum.dialogue_id for datum in train_data})  # type: ignore
        random.shuffle(train_dialogue_ids)
        test_data = dev_data
        num_train_dialogues = len(train_dialogue_ids)
        dev_dialogue_ids = set(train_dialogue_ids[: int(0.1 * num_train_dialogues)])
        dev_data = [
            datum for datum in train_data if datum.dialogue_id in dev_dialogue_ids
        ]
        train_data = [
            datum for datum in train_data if datum.dialogue_id not in dev_dialogue_ids
        ]

    print(
        f"Input sizes for creating benchclamp splits: "
        f"Train {len(train_data)}, Dev {len(dev_data)}, Test {len(test_data)}"
    )
    train_turn_ids = [
        (datum.dialogue_id, datum.turn_part_index) for datum in train_data
    ]
    dev_turn_ids = [(datum.dialogue_id, datum.turn_part_index) for datum in dev_data]
    test_turn_ids = [(datum.dialogue_id, datum.turn_part_index) for datum in test_data]
    assert (
        len(set(train_turn_ids)) == len(train_turn_ids)
        and len(set(dev_turn_ids)) == len(dev_turn_ids)
        and len(set(test_turn_ids)) == len(test_turn_ids)
    ), "Multiple data points have same data id, make sure all input data have unique data ids."

    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl_file(test_data, str(output_dir / "test_all.jsonl"))
    save_jsonl_file(gather_subset(test_data, 2000), str(output_dir / "test.jsonl"))

    save_jsonl_file(dev_data, str(output_dir / "dev_all.jsonl"))
    save_jsonl_file(gather_subset(dev_data, 500), str(output_dir / "dev_medium.jsonl"))
    save_jsonl_file(gather_subset(dev_data, 50), str(output_dir / "dev_low.jsonl"))

    save_jsonl_file(train_data, str(output_dir / "train_all.jsonl"))
    for split_size_category, num_splits, train_split_size in [
        ("low", 5, 500),
        ("medium", 3, 5000),
    ]:
        for split_id in range(num_splits):
            train_subset_file = (
                output_dir / f"train_{split_size_category}_{split_id}.jsonl"
            )
            train_subset = gather_subset(train_data, train_split_size)
            save_jsonl_file(train_subset, str(train_subset_file))
