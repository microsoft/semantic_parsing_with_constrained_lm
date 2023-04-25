# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, TextIO, Tuple

import jsons
from blobfile import BlobFile

from semantic_parsing_with_constrained_lm.datum import BenchClampDatum, FullDatum
from semantic_parsing_with_constrained_lm.domains.lispress_v2.sequence_creator import (
    LastAgentUtterance,
    LastUserAgentUtterance,
    OracleRewrittenUtterance,
    LastPlan,
    RewrittenUtterance
)
from semantic_parsing_with_constrained_lm.domains.sql.sequence_creator import CoSqlUtterance
from semantic_parsing_with_constrained_lm.domains.sql.sql_datum import SqlDatum
from semantic_parsing_with_constrained_lm.paths import BENCH_CLAMP_PROCESSED_DATA_DIR_AZURE
from semantic_parsing_with_constrained_lm.sequence_creator import (
    IdentitySequenceCreator,
    SequenceCreator,
)
from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer


def data_from_textio(data_file: TextIO) -> List[BenchClampDatum]:
    return [jsons.loads(line.strip(), cls=BenchClampDatum) for line in data_file]


@dataclass  # type: ignore
class ClampDataConfig(abc.ABC):
    data_id: str
    split_name: str
    output_type: Optional[str] = None
    domain: Optional[str] = None
    tokenizer: Optional[ClampTokenizer] = None
    input_sequence_creator: Optional[SequenceCreator] = None

    def __post_init__(self):
        if self.input_sequence_creator is None:
            self.input_sequence_creator = IdentitySequenceCreator()

    @abstractmethod
    def setup_data(self) -> Tuple[List[FullDatum], List[FullDatum], List[FullDatum]]:
        pass

    def modify_data_with_sequence_creator(
        self, data: List[BenchClampDatum]
    ) -> List[FullDatum]:
        data_for_expt = []
        for datum in data:
            input_sequence = (
                self.input_sequence_creator.create_sequence(datum)
                if self.input_sequence_creator is not None
                else datum.utterance
            )
            if datum.schema_name is not None:
                data_for_expt.append(
                    SqlDatum(
                        dialogue_id=datum.dialogue_id,
                        turn_part_index=datum.turn_part_index,
                        natural=input_sequence,
                        canonical=datum.plan,
                        agent_context="",
                        schema_name=datum.schema_name,
                    )
                )
            else:
                data_for_expt.append(
                    FullDatum(
                        dialogue_id=datum.dialogue_id,
                        turn_part_index=datum.turn_part_index,
                        natural=input_sequence,
                        canonical=datum.plan,
                        agent_context="",
                    )
                )
        return data_for_expt


@dataclass
class BenchClampDatasetConfig(ClampDataConfig):
    dataset_name: str = ""
    eval_on_full_test: bool = False
    merge_train_and_dev: bool = False

    def setup_data(self) -> Tuple[List[FullDatum], List[FullDatum], List[FullDatum]]:
        domain_str = self.domain + "/" if self.domain is not None else ""
        if "low" in self.split_name:
            dev_data_suffix = "low"
        else:
            dev_data_suffix = "medium"
        train_data_file = f"{BENCH_CLAMP_PROCESSED_DATA_DIR_AZURE}/{self.dataset_name}/{domain_str}train_{self.split_name}.jsonl"
        dev_data_file = f"{BENCH_CLAMP_PROCESSED_DATA_DIR_AZURE}/{self.dataset_name}/{domain_str}dev_{dev_data_suffix}.jsonl"
        if self.eval_on_full_test:
            test_data_file = f"{BENCH_CLAMP_PROCESSED_DATA_DIR_AZURE}/{self.dataset_name}/{domain_str}test_all.jsonl"
        else:
            test_data_file = f"{BENCH_CLAMP_PROCESSED_DATA_DIR_AZURE}/{self.dataset_name}/{domain_str}test.jsonl"

        if self.dataset_name == "CalFlowFindEvent":
            '''
            train_data_file = "https://internshipsalience2022.blob.core.windows.net/internship-salience-2022/find_event.train.jsonl"
            dev_data_file = "https://internshipsalience2022.blob.core.windows.net/internship-salience-2022/find_event.valid.jsonl" 
            test_data_file = "https://internshipsalience2022.blob.core.windows.net/internship-salience-2022/find_event.valid.jsonl" 
            '''
            train_data_file = "./data/find_event.train.jsonl"
            dev_data_file = "./data/find_event.valid.jsonl" 
            test_data_file = "./data/find_event.valid.jsonl" 

        if self.dataset_name == "CalFlowFindEventRevise":
            '''
            train_data_file = "https://internshipsalience2022.blob.core.windows.net/internship-salience-2022/find_event.train.jsonl"
            dev_data_file = "https://internshipsalience2022.blob.core.windows.net/internship-salience-2022/find_event.valid.jsonl" 
            test_data_file = "https://internshipsalience2022.blob.core.windows.net/internship-salience-2022/find_event.valid.jsonl" 
            '''
            train_data_file = "./data/02-04-2023-find_event_revise.train.annotation_split.jsonl"
            dev_data_file = "./data/02-04-2023-find_event_revise.valid.annotation_split.text-davinci-003.jsonl" 
            test_data_file = "./data/02-04-2023-find_event_revise.valid.annotation_split.text-davinci-003.jsonl"  


        with BlobFile(str(train_data_file)) as bf:
            print(f"Reading {train_data_file}")
            train_data = data_from_textio(bf)
        with BlobFile(str(dev_data_file)) as bf:
            print(f"Reading {dev_data_file}")
            dev_data = data_from_textio(bf)
        with BlobFile(str(test_data_file)) as bf:
            print(f"Reading {test_data_file}")
            test_data = data_from_textio(bf)
        if self.merge_train_and_dev:
            train_data.extend(dev_data)

        return (
            self.modify_data_with_sequence_creator(train_data),
            self.modify_data_with_sequence_creator(dev_data),
            self.modify_data_with_sequence_creator(test_data),
        )


class BenchClampDataset(str, Enum):
    CalFlowV2 = "CalFlowV2"
    TreeDST = "TreeDST"
    Overnight = "Overnight"
    MTOP = "MTOP"
    Spider = "Spider"
    CoSQL = "CoSQL"
    CalFlowFindEvent = "CalFlowFindEvent"
    CalFlowFindEventRevise = "CalFlowFindEventRevise"


OVERNIGHT_DOMAINS = [
    "basketball",
    "blocks",
    "calendar",
    "housing",
    "publications",
    "recipes",
    "restaurants",
    "socialnetwork",
]

MTOP_LANGUAGES = ["de", "en", "es", "fr", "hi", "th"]

BENCHCLAMP_SPLIT_NAMES: List[str] = (
    [f"low_{i}" for i in range(3)] + [f"medium_{i}" for i in range(1)] + ["all"]
)

BENCHCLAMP_DATA_CONFIGS: List[ClampDataConfig] = (
    [
        BenchClampDatasetConfig(
            data_id=f"calflow_{input_sequence_creator_name}_{split_name}",
            split_name=split_name,
            domain=None,
            dataset_name=BenchClampDataset.CalFlowV2.value,
            input_sequence_creator=input_sequence_creator,
        )
        for input_sequence_creator_name, input_sequence_creator, split_names in [
            ("no_context", IdentitySequenceCreator(), BENCHCLAMP_SPLIT_NAMES),
            ("last_agent", LastAgentUtterance(), BENCHCLAMP_SPLIT_NAMES),
            ("last_user", LastUserAgentUtterance(), BENCHCLAMP_SPLIT_NAMES),
            ("oracle_rewritten_user", OracleRewrittenUtterance(), BENCHCLAMP_SPLIT_NAMES),
            ("rewritten_user", RewrittenUtterance(), BENCHCLAMP_SPLIT_NAMES),
            ("last_plan", LastPlan(), BENCHCLAMP_SPLIT_NAMES),

        ]
        for split_name in split_names
    ] +
    [
        BenchClampDatasetConfig(
            data_id=f"calflowfindevent_{input_sequence_creator_name}_{split_name}",
            split_name=split_name,
            domain=None,
            dataset_name=BenchClampDataset.CalFlowFindEvent.value,
            input_sequence_creator=input_sequence_creator,
        )
        for input_sequence_creator_name, input_sequence_creator, split_names in [
            ("no_context", IdentitySequenceCreator(), BENCHCLAMP_SPLIT_NAMES),
            ("last_agent", LastAgentUtterance(), BENCHCLAMP_SPLIT_NAMES),
            ("last_user", LastUserAgentUtterance(), BENCHCLAMP_SPLIT_NAMES),
            ("oracle_rewritten_user", OracleRewrittenUtterance(), BENCHCLAMP_SPLIT_NAMES),
            ("rewritten_user", RewrittenUtterance(), BENCHCLAMP_SPLIT_NAMES),
            ("last_plan", LastPlan(), BENCHCLAMP_SPLIT_NAMES),

        ]
        for split_name in split_names
    ] +
    [
        BenchClampDatasetConfig(
            data_id=f"calflowfindeventrevise_{input_sequence_creator_name}_{split_name}",
            split_name=split_name,
            domain=None,
            dataset_name=BenchClampDataset.CalFlowFindEventRevise.value,
            input_sequence_creator=input_sequence_creator,
        )
        for input_sequence_creator_name, input_sequence_creator, split_names in [
            ("no_context", IdentitySequenceCreator(), BENCHCLAMP_SPLIT_NAMES),
            ("last_agent", LastAgentUtterance(), BENCHCLAMP_SPLIT_NAMES),
            ("last_user", LastUserAgentUtterance(), BENCHCLAMP_SPLIT_NAMES),
            ("oracle_rewritten_user", OracleRewrittenUtterance(), BENCHCLAMP_SPLIT_NAMES),
            ("rewritten_user", RewrittenUtterance(), BENCHCLAMP_SPLIT_NAMES),
            ("last_plan", LastPlan(), BENCHCLAMP_SPLIT_NAMES),
        ]
        for split_name in split_names
    ]
    + [
        BenchClampDatasetConfig(
            data_id=f"tree_dst_{input_sequence_creator_name}_{split_name}",
            split_name=split_name,
            domain=None,
            dataset_name=BenchClampDataset.TreeDST.value,
            input_sequence_creator=input_sequence_creator,
        )
        for input_sequence_creator_name, input_sequence_creator, split_names in [
            ("no_context", IdentitySequenceCreator(), BENCHCLAMP_SPLIT_NAMES),
            ("last_agent", LastAgentUtterance(), ["low_0", "low_1", "low_2"]),
            ("last_user", LastUserAgentUtterance(), ["medium_0", "all"]),
        ]
        for split_name in split_names
    ]
    + [
        BenchClampDatasetConfig(
            data_id=f"mtop_{language}_no_context_{split_name}",
            split_name=split_name,
            domain=language,
            dataset_name=BenchClampDataset.MTOP.value,
        )
        for split_name in BENCHCLAMP_SPLIT_NAMES
        for language in ["en"]
    ]
    + [  # type: ignore
        BenchClampDatasetConfig(
            data_id=f"overnight_{domain}_no_context_{split_name}",
            split_name=split_name,
            domain=domain,
            dataset_name=BenchClampDataset.Overnight.value,
        )
        for split_name in BENCHCLAMP_SPLIT_NAMES
        for domain in ["blocks"]
    ]
    + [
        BenchClampDatasetConfig(
            data_id=f"spider_past_none_db_val_{split_name}",
            split_name=split_name,
            dataset_name=BenchClampDataset.Spider.value,
            input_sequence_creator=CoSqlUtterance(
                use_db_val=True, past_utterances="none"
            ),
        )
        for split_name in BENCHCLAMP_SPLIT_NAMES
    ]
    + [
        BenchClampDatasetConfig(
            data_id=f"cosql_{input_sequence_creator_name}_{split_name}",
            split_name=split_name,
            dataset_name=BenchClampDataset.CoSQL.value,
            input_sequence_creator=input_sequence_creator,
        )
        for input_sequence_creator_name, input_sequence_creator, split_names in [
            (
                "past_one_db_val",
                CoSqlUtterance(use_db_val=True, past_utterances="one"),
                ["low_0", "low_1", "low_2"],
            ),
            (
                "past_all_db_val",
                CoSqlUtterance(use_db_val=True, past_utterances="all"),
                ["medium_0", "all"],
            ),
        ]
        for split_name in split_names
    ]
    + [
        BenchClampDatasetConfig(
            data_id="spider_past_none_db_val_all_merge_train_dev",
            split_name="all",
            dataset_name=BenchClampDataset.Spider.value,
            input_sequence_creator=CoSqlUtterance(
                use_db_val=True, past_utterances="none"
            ),
            merge_train_and_dev=True,
        ),
        BenchClampDatasetConfig(
            data_id="cosql_past_all_db_val_all_merge_train_dev",
            split_name="all",
            dataset_name=BenchClampDataset.CoSQL.value,
            input_sequence_creator=CoSqlUtterance(
                use_db_val=True, past_utterances="all"
            ),
            merge_train_and_dev=True,
        ),
    ]
)
