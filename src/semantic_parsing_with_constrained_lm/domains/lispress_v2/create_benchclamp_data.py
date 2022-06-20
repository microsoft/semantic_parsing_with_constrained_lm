# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
import json
import sys
from pathlib import Path
from typing import Optional, Set

import jsons
from blobfile import BlobFile
from dataflow.core.io_utils import load_jsonl_file
from transformers import GPT2Tokenizer

from semantic_parsing_with_constrained_lm.earley.cfg import load_grammar_from_directory
from semantic_parsing_with_constrained_lm.datum import BenchClampDatum, FullDatum
from semantic_parsing_with_constrained_lm.domains.benchclamp_data_setup import BenchClampDataset
from semantic_parsing_with_constrained_lm.domains.create_benchclamp_splits import (
    can_force_decode,
    create_benchclamp_splits,
)
from semantic_parsing_with_constrained_lm.domains.lispress_v2.grammar import (
    create_partial_parse_builder,
    extract_grammar,
)
from semantic_parsing_with_constrained_lm.domains.lispress_v2.lispress_exp import DialogueV2
from semantic_parsing_with_constrained_lm.paths import (
    BENCH_CLAMP_GRAMMAR_DATA_DIR,
    BENCH_CLAMP_PROCESSED_DATA_DIR,
    BENCH_CLAMP_RAW_DATA_DIR,
)
from semantic_parsing_with_constrained_lm.tokenization import GPT2ClampTokenizer
from semantic_parsing_with_constrained_lm.finetune.platypus import calflow_to_datum_format


def extract_and_write_grammar(
    train_dataflow_dialogues_jsonl: Path,
    grammar_output_dir: Path,
    whitelisted_dialogue_ids: Optional[Set[str]] = None,
) -> None:
    # Extract grammar
    print("Extracting grammar ...")
    sys.setrecursionlimit(50000)
    dataflow_dialogues = load_jsonl_file(
        data_jsonl=str(train_dataflow_dialogues_jsonl),
        cls=DialogueV2,
        unit=" dialogues",
    )
    grammar_rules = extract_grammar(
        dataflow_dialogues, whitelisted_dialogue_ids=whitelisted_dialogue_ids
    )
    # Write Grammar
    print("Writing grammar to file ...")
    grammar_output_dir.mkdir(exist_ok=True, parents=True)  # type: ignore
    with open(grammar_output_dir / "generated.cfg", "w") as fp:  # type: ignore
        fp.write("\n".join(sorted(grammar_rules)) + "\n")


def write_data_and_grammar(
    train_dataflow_dialogues_jsonl: Path,
    dev_dataflow_dialogues_jsonl: Path,
    test_dataflow_dialogues_jsonl: Optional[Path],
    datum_output_dir: Path,
    grammar_output_dir: Optional[Path] = None,
) -> None:
    """
    Converts dialogues from dataflow_dialogues_jsonl to FullDatum format and writes to datum_format_file.
    If grammar_output_dir is set, extracts grammar and writes it to grammar_output_dir.
    """
    # Create data splits
    print("Creating data splits ...")
    train_data = calflow_to_datum_format(str(train_dataflow_dialogues_jsonl))
    dev_data = calflow_to_datum_format(str(dev_dataflow_dialogues_jsonl))
    train_benchclamp_data = []
    dev_benchclamp_data = []
    for data, benchclamp_data in [
        (train_data, train_benchclamp_data),
        (dev_data, dev_benchclamp_data),
    ]:
        for datum in data:
            context = json.loads(datum.agent_context)  # type: ignore
            benchclamp_data.append(
                BenchClampDatum(
                    dialogue_id=datum.dialogue_id,
                    turn_part_index=datum.turn_part_index,
                    utterance=datum.natural,
                    plan=datum.canonical,
                    last_plan=context["plan"],
                    last_user_utterance=context["user_utterance"],
                    last_agent_utterance=context["agent_utterance"],
                )
            )

    test_benchclamp_data = None
    if test_dataflow_dialogues_jsonl is not None:
        test_data = calflow_to_datum_format(str(test_dataflow_dialogues_jsonl))
        test_benchclamp_data = []
        for datum in test_data:
            context = json.loads(datum.agent_context)  # type: ignore
            test_benchclamp_data.append(
                BenchClampDatum(
                    dialogue_id=datum.dialogue_id,
                    turn_part_index=datum.turn_part_index,
                    utterance=datum.natural,
                    plan=datum.canonical,
                    last_plan=context["plan"],
                    last_user_utterance=context["user_utterance"],
                    last_agent_utterance=context["agent_utterance"],
                )
            )

    create_benchclamp_splits(
        train_benchclamp_data,
        dev_benchclamp_data,
        test_benchclamp_data,
        datum_output_dir,
    )

    extract_and_write_grammar(
        train_dataflow_dialogues_jsonl,
        grammar_output_dir,  # type: ignore
        whitelisted_dialogue_ids=None,
    )

    print("Testing ...")
    clamp_tokenizer = GPT2ClampTokenizer(GPT2Tokenizer.from_pretrained("gpt2"))
    partial_parse_builder = create_partial_parse_builder(
        load_grammar_from_directory(str(grammar_output_dir)), clamp_tokenizer
    )
    total = 0
    wrong = 0
    print("Testing if force decoding possible for first 1000 train dialogues")
    for dataflow_dialogue in itertools.islice(
        load_jsonl_file(
            data_jsonl=str(train_dataflow_dialogues_jsonl),
            cls=DialogueV2,
            unit=" dialogues",
        ),
        1000,
    ):
        for _, turn in enumerate(dataflow_dialogue.turns):
            total += 1
            if not can_force_decode(
                clamp_tokenizer,
                partial_parse_builder,
                FullDatum(
                    dialogue_id="",
                    turn_part_index=0,
                    natural=turn.user_utterance.original_text,
                    canonical=turn.lispress,
                    agent_context="",
                ),
            ):
                wrong += 1

    print(f"Force Decode Errors %: {wrong} / {total}")


def create_grammar_from_train_split():
    for dataset, train_data_file in [
        (
            BenchClampDataset.CalFlowV2.value,
            BENCH_CLAMP_RAW_DATA_DIR
            / BenchClampDataset.CalFlowV2.value
            / "train.dataflow_dialogues.jsonl",
        ),
        (
            BenchClampDataset.TreeDST.value,
            BENCH_CLAMP_RAW_DATA_DIR
            / BenchClampDataset.TreeDST.value
            / "train_dst.dataflow_dialogues.jsonl",
        ),
    ]:
        for split_name in ["low_0", "low_1", "low_2", "medium_0"]:
            grammar_output_dir = BENCH_CLAMP_GRAMMAR_DATA_DIR / dataset / split_name
            with BlobFile(
                f"https://benchclamp.blob.core.windows.net/wip/benchclamp/processed/"
                f"{dataset}/train_{split_name}.jsonl"
            ) as bf:
                whitelisted_dialogue_ids = set()
                for line in bf:
                    dialogue = jsons.loads(line.strip(), cls=BenchClampDatum)
                    whitelisted_dialogue_ids.add(dialogue.dialogue_id)
            print(f"Running grammar extraction {dataset} {split_name}")
            extract_and_write_grammar(
                train_data_file,
                grammar_output_dir,
                whitelisted_dialogue_ids=whitelisted_dialogue_ids,
            )


def main():
    write_data_and_grammar(
        train_dataflow_dialogues_jsonl=BENCH_CLAMP_RAW_DATA_DIR
        / f"{BenchClampDataset.CalFlowV2}/train.dataflow_dialogues.jsonl",
        dev_dataflow_dialogues_jsonl=BENCH_CLAMP_RAW_DATA_DIR
        / f"{BenchClampDataset.CalFlowV2}/valid.dataflow_dialogues.jsonl",
        test_dataflow_dialogues_jsonl=None,
        datum_output_dir=BENCH_CLAMP_PROCESSED_DATA_DIR
        / f"{BenchClampDataset.CalFlowV2}/",
        grammar_output_dir=BENCH_CLAMP_GRAMMAR_DATA_DIR
        / f"{BenchClampDataset.CalFlowV2}/",
    )
    write_data_and_grammar(
        train_dataflow_dialogues_jsonl=BENCH_CLAMP_RAW_DATA_DIR
        / f"{BenchClampDataset.TreeDST}/train_dst.dataflow_dialogues.jsonl",
        dev_dataflow_dialogues_jsonl=BENCH_CLAMP_RAW_DATA_DIR
        / f"{BenchClampDataset.TreeDST}/dev_dst.dataflow_dialogues.jsonl",
        test_dataflow_dialogues_jsonl=BENCH_CLAMP_RAW_DATA_DIR
        / f"{BenchClampDataset.TreeDST}/test_dst.dataflow_dialogues.jsonl",
        datum_output_dir=BENCH_CLAMP_PROCESSED_DATA_DIR
        / f"{BenchClampDataset.TreeDST}/",
        grammar_output_dir=BENCH_CLAMP_GRAMMAR_DATA_DIR
        / f"{BenchClampDataset.TreeDST}/",
    )


if __name__ == "__main__":
    main()
