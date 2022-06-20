# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List

from blobfile import BlobFile

from semantic_parsing_with_constrained_lm.util.trie import Trie
from semantic_parsing_with_constrained_lm.util.types import StrPath
from semantic_parsing_with_constrained_lm.datum import Datum, FullDatum
from semantic_parsing_with_constrained_lm.decoding.trie_partial_parse import TriePartialParse
from semantic_parsing_with_constrained_lm.domains.calflow.write_data import CACHE_DIR
from semantic_parsing_with_constrained_lm.eval import TopKExactMatch
from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer


class OutputType(str, Enum):
    Utterance = "utterance"
    MeaningRepresentation = "meaningRepresentation"


@dataclass
class TopKDenotationMatch(TopKExactMatch[FullDatum]):
    canonical_to_denotation: Dict[str, str]

    def _is_correct(self, pred: str, datum: FullDatum) -> bool:
        target = datum.canonical
        pred_denotation = self.canonical_to_denotation.get(pred)
        target_denotation = self.canonical_to_denotation.get(target, None)
        if pred_denotation is None and target_denotation is None:
            return pred == target
        else:
            return pred_denotation == target_denotation


@dataclass
class OvernightPieces:
    train_data: List[FullDatum]
    test_data: List[FullDatum]
    partial_parse_builder: Callable[[Datum], TriePartialParse]
    denotation_metric: TopKDenotationMatch
    max_length: int

    @staticmethod
    def from_dir(
        tokenizer: ClampTokenizer,
        root_dir: StrPath,
        domain: str,
        is_dev: bool,
        k: int,
        output_type: OutputType = OutputType.Utterance,
        simplify_logical_forms=False,
        prefix_with_space=False,
    ) -> "OvernightPieces":
        data_pieces = OvernightDataPieces.from_dir(
            root_dir, domain, is_dev, output_type, simplify_logical_forms
        )
        decoder_pieces = OvernightDecoderPieces.create(
            data_pieces, tokenizer, k, prefix_with_space
        )

        return OvernightPieces(
            data_pieces.train_data,
            data_pieces.test_data,
            # https://github.com/python/mypy/issues/5485
            decoder_pieces.partial_parse_builder,  # type: ignore
            decoder_pieces.denotation_metric,
            decoder_pieces.max_length,
        )


@dataclass
class OvernightDataPieces:
    train_data: List[FullDatum]
    test_data: List[FullDatum]
    target_output_to_denotation: Dict[str, str]

    @staticmethod
    def from_dir(
        root_dir: StrPath,
        domain: str,
        is_dev: bool,
        output_type: OutputType = OutputType.MeaningRepresentation,
        simplify_logical_forms: bool = False,
    ) -> "OvernightDataPieces":
        # TODO make this configurable?
        with BlobFile(str(root_dir) + f"/{domain}.canonical.json") as bf:
            canonical_data = json.load(bf)

        if output_type == OutputType.Utterance:
            target_output_to_denotation = {
                k: v["denotation"] for k, v in canonical_data.items()
            }
            datum_key = "canonical"
        elif output_type == OutputType.MeaningRepresentation:
            target_output_to_denotation = {}
            for program_info in canonical_data.values():
                formula = program_info["formula"]
                if formula is None:
                    continue
                if simplify_logical_forms:
                    formula = OvernightDataPieces.simplify_lf(formula)
                assert formula not in target_output_to_denotation
                target_output_to_denotation[formula] = program_info["denotation"]
            datum_key = "formula"
        else:
            raise ValueError(output_type)

        train_data, test_data = [
            [
                FullDatum(
                    dialogue_id=f"{dataset_name}-{i}",
                    turn_part_index=None,
                    agent_context=None,
                    natural=d["natural"],
                    canonical=OvernightDataPieces.simplify_lf(d[datum_key])
                    if simplify_logical_forms
                    else d[datum_key],
                )
                for i, line in enumerate(
                    BlobFile(path, streaming=False, cache_dir=CACHE_DIR)
                )
                for d in [json.loads(line)]
            ]
            for dataset_name, path in (
                (
                    "train",
                    f"{root_dir}/{domain}.train_with{'out' if is_dev else ''}_dev.jsonl",
                ),
                ("eval", f"{root_dir}/{domain}.{'dev' if is_dev else 'test'}.jsonl"),
            )
        ]

        return OvernightDataPieces(train_data, test_data, target_output_to_denotation)

    @staticmethod
    def simplify_lf(lf: str) -> str:
        return lf.replace("edu.stanford.nlp.sempre.overnight.SimpleWorld.", "")


@dataclass
class OvernightDecoderPieces:
    data_pieces: OvernightDataPieces
    partial_parse_builder: Callable[[Datum], TriePartialParse]
    denotation_metric: TopKDenotationMatch
    max_length: int

    @staticmethod
    def create(
        data_pieces: OvernightDataPieces,
        tokenizer: ClampTokenizer,
        k: int,
        prefix_with_space: bool = False,
    ) -> "OvernightDecoderPieces":
        if prefix_with_space:
            canonical_trie = Trie(
                tokenizer.encode(" " + canon)
                for canon in data_pieces.target_output_to_denotation
            )
        else:
            canonical_trie = Trie(
                tokenizer.encode(canon)
                for canon in data_pieces.target_output_to_denotation
            )
        partial_parse_builder = lambda _: TriePartialParse(canonical_trie)

        denotation_metric = TopKDenotationMatch(
            k, data_pieces.target_output_to_denotation
        )
        max_length = max(len(x) for x in canonical_trie)

        return OvernightDecoderPieces(
            data_pieces, partial_parse_builder, denotation_metric, max_length
        )
