# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from transformers import GPT2Tokenizer

from semantic_parsing_with_constrained_lm.earley.cfg import load_grammar_from_directory
from semantic_parsing_with_constrained_lm.datum import BenchClampDatum, FullDatum
from semantic_parsing_with_constrained_lm.domains.benchclamp_data_setup import (
    MTOP_LANGUAGES,
    BenchClampDataset,
)
from semantic_parsing_with_constrained_lm.domains.create_benchclamp_splits import (
    can_force_decode,
    create_benchclamp_splits,
)
from semantic_parsing_with_constrained_lm.domains.mtop.grammar import create_partial_parse_builder
from semantic_parsing_with_constrained_lm.paths import (
    BENCH_CLAMP_GRAMMAR_DATA_DIR,
    BENCH_CLAMP_PROCESSED_DATA_DIR,
    BENCH_CLAMP_RAW_DATA_DIR,
)
from semantic_parsing_with_constrained_lm.tokenization import GPT2ClampTokenizer


@dataclass
class MTOPExpr:
    label: str
    span: Optional[str] = None
    children: List["MTOPExpr"] = dataclasses.field(default_factory=list)

    def __repr__(self) -> str:
        if self.span is None and len(self.children) == 0:
            return f"[{self.label} ]"
        elif self.span is not None:
            return f"[{self.label} {self.span} ]"
        else:
            return (
                f"[{self.label} "
                + " ".join([str(child) for child in self.children])
                + " ]"
            )


def parse_mtop_expression(mtop_str: str, slot_to_span_map: Dict[str, str]) -> MTOPExpr:
    """
    Parses MTOP expressions into structured output.
    An example MTOP expression shown below:
      [IN:GET_MESSAGE [SL:CONTACT Angelika Kratzer ] [SL:TYPE_CONTENT video ] [SL:RECIPIENT me ] ]
    """
    mtop_str = mtop_str.strip()
    assert mtop_str[0] == "[" and mtop_str[-1] == "]"
    mtop_str = mtop_str[1:-1]
    label = mtop_str.split()[0]
    if "[" in mtop_str and "]" in mtop_str:
        # Not a leaf node, so won't have span
        open_brackets = 0
        start_sub_expr_index = -1

        children = []
        for index, ch in enumerate(mtop_str):
            if ch == "[":
                open_brackets += 1
                if open_brackets == 1:
                    start_sub_expr_index = index
            elif ch == "]":
                open_brackets -= 1
                if open_brackets == 0:
                    children.append(
                        parse_mtop_expression(
                            mtop_str[start_sub_expr_index : index + 1], slot_to_span_map
                        )
                    )

        return MTOPExpr(label=label, children=children)
    else:
        span = None
        if label in slot_to_span_map:
            span = slot_to_span_map[label]

        return MTOPExpr(label=label, span=span)


def get_nt_from_label(label: str) -> str:
    return label.replace(":", "_") + "_NT"


def extract_grammar_rules(mtop_expr: MTOPExpr) -> Set[str]:
    lhs = get_nt_from_label(mtop_expr.label)
    rules = set()
    if mtop_expr.span is not None:
        rhs = f'"[{mtop_expr.label} " any_char_star " ]"'
        rules.add(f"{lhs} -> {rhs}")
        rhs = f'"[{mtop_expr.label} ]"'
        rules.add(f"{lhs} -> {rhs}")
    else:
        rhs_items = [get_nt_from_label(expr.label) for expr in mtop_expr.children]
        rhs = ' " " '.join([f'"[{mtop_expr.label}"'] + rhs_items + ['"]"'])
        rules.add(f"{lhs} -> {rhs}")
        for expr in mtop_expr.children:
            rules.update(extract_grammar_rules(expr))

    return rules


def mtop_to_datum_format(
    mtop_tsv_file: Path,
) -> List[Tuple[BenchClampDatum, Dict[str, str]]]:
    data = []
    with open(mtop_tsv_file, "r") as fp:
        for line in fp.readlines():
            items = line.split("\t")
            idx = items[0]
            utterance = items[3]
            meaning = items[6]
            slot_spans_str = re.split(",|，", items[2])
            slot_to_span_map = {}
            for slot_span_str in slot_spans_str:
                if slot_span_str.strip() == "":
                    continue
                start, end, slot_part_1, slot_part_2 = slot_span_str.split(":")
                slot_to_span_map[slot_part_1 + ":" + slot_part_2] = utterance[
                    int(start) : int(end)
                ]
            data.append(
                (
                    BenchClampDatum(
                        dialogue_id=idx,
                        turn_part_index=0,
                        utterance=utterance,
                        plan=meaning,
                    ),
                    slot_to_span_map,
                )
            )

    return data


def main():
    for language in MTOP_LANGUAGES:
        train_tsv = (
            BENCH_CLAMP_RAW_DATA_DIR / f"{BenchClampDataset.MTOP}/{language}/train.txt"
        )
        dev_tsv = (
            BENCH_CLAMP_RAW_DATA_DIR / f"{BenchClampDataset.MTOP}/{language}/eval.txt"
        )
        test_tsv = (
            BENCH_CLAMP_RAW_DATA_DIR / f"{BenchClampDataset.MTOP}/{language}/test.txt"
        )
        train_data_with_slot_spans = mtop_to_datum_format(train_tsv)
        dev_data_with_slot_spans = mtop_to_datum_format(dev_tsv)
        test_data_with_slot_spans = mtop_to_datum_format(test_tsv)
        train_data = []
        dev_data = []
        test_data = []
        print("Extracting grammar ...")
        grammar_rules = set()
        for datum, slot_span_map in train_data_with_slot_spans:
            mtop_expr = parse_mtop_expression(datum.plan, slot_span_map)
            grammar_rules.update(extract_grammar_rules(mtop_expr))
            root_type_nt = get_nt_from_label(mtop_expr.label)
            grammar_rules.add(f'start -> " " {root_type_nt}')
            train_data.append(dataclasses.replace(datum, plan=str(mtop_expr)))

        for datum, slot_span_map in dev_data_with_slot_spans:
            mtop_expr = parse_mtop_expression(datum.plan, slot_span_map)
            dev_data.append(dataclasses.replace(datum, plan=str(mtop_expr)))

        for datum, slot_span_map in test_data_with_slot_spans:
            mtop_expr = parse_mtop_expression(datum.plan, slot_span_map)
            test_data.append(dataclasses.replace(datum, plan=str(mtop_expr)))

        grammar_rules.add(r"any_char_star -> [\u0001-\U0010FFFF]*")

        # Write Grammar
        print("Writing grammar to file ...")
        grammar_dir = (
            BENCH_CLAMP_GRAMMAR_DATA_DIR / f"{BenchClampDataset.MTOP}/{language}"
        )
        grammar_dir.mkdir(exist_ok=True, parents=True)
        with open(grammar_dir / "generated.cfg", "w") as fp:
            fp.write("\n".join(sorted(grammar_rules)) + "\n")

        print("Creating data splits ...")
        create_benchclamp_splits(
            train_data,
            dev_data,
            test_data,
            BENCH_CLAMP_PROCESSED_DATA_DIR / f"{BenchClampDataset.MTOP}/{language}/",
        )

        print("Testing ...")
        clamp_tokenizer = GPT2ClampTokenizer(GPT2Tokenizer.from_pretrained("gpt2"))
        partial_parse_builder = create_partial_parse_builder(
            load_grammar_from_directory(str(grammar_dir)), clamp_tokenizer
        )
        total = 0
        wrong = 0
        print("Testing if force decoding possible for first 1000 train examples")
        for datum in itertools.islice(train_data, 1000):
            total += 1
            if not can_force_decode(
                clamp_tokenizer,
                partial_parse_builder,
                FullDatum(
                    dialogue_id="",
                    turn_part_index=0,
                    natural=datum.utterance,
                    canonical=datum.plan,
                    agent_context="",
                ),
            ):
                wrong += 1

        print(f"Force Decode Errors %: {wrong} / {total}")


if __name__ == "__main__":
    main()
