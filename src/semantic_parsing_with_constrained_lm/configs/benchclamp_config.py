# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Config to run training and evaluation experiments with BenchCLAMP with non-GPT-3 language models.
"""
import pdb 
import dataclasses
import functools
import json
import sys
import os 
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import torch
from transformers import IntervalStrategy, TrainingArguments
from typing_extensions import Literal

from semantic_parsing_with_constrained_lm.configs.lib.benchclamp import (
    COSQL_TABLES_FILE,
    SPIDER_TABLES_FILE,
    TEST_SUITE_DATABASE_PATH,
    TEST_SUITE_PATH,
    create_partial_parse_builder,
)
from semantic_parsing_with_constrained_lm.configs.lib.common import make_semantic_parser
from semantic_parsing_with_constrained_lm.model import ProblemFactory, DecodingSetup
from semantic_parsing_with_constrained_lm.datum import Datum, FullDatum
from semantic_parsing_with_constrained_lm.decoding.partial_parse import (
    PartialParse,
    StartsWithSpacePartialParse,
)
from semantic_parsing_with_constrained_lm.domains.benchclamp_data_setup import (
    BENCHCLAMP_DATA_CONFIGS,
    BenchClampDataset,
    BenchClampDatasetConfig,
    ClampDataConfig,
)

from semantic_parsing_with_constrained_lm.domains.lispress_v2.lispress_exp import TopKLispressMatch
from semantic_parsing_with_constrained_lm.domains.overnight import OutputType, OvernightPieces
from semantic_parsing_with_constrained_lm.domains.sql.sql_metric import SQLTestSuiteMatch
from semantic_parsing_with_constrained_lm.eval import Metric, TopKExactMatch
from semantic_parsing_with_constrained_lm.fit_max_steps import compute_and_print_fit
from semantic_parsing_with_constrained_lm.lm import Seq2SeqModel
from semantic_parsing_with_constrained_lm.lm_bart import Seq2SeqBart
from semantic_parsing_with_constrained_lm.lm_gpt2 import Seq2SeqGPT2
from semantic_parsing_with_constrained_lm.paths import OVERNIGHT_DATA_DIR_AZURE, RUN_ON_AML
from semantic_parsing_with_constrained_lm.run_exp import Experiment
from semantic_parsing_with_constrained_lm.tokenization import T5ClampTokenizer
from semantic_parsing_with_constrained_lm.train_model_setup import (
    BartModelConfig,
    ClampModelConfig,
    CodeT5ModelConfig,
    GPT2ModelConfig,
    T5ModelConfig,
)
from semantic_parsing_with_constrained_lm.finetune.lm_finetune import TrainExperiment

# /mnt/my_input and /mnt/my_output refers to location names used in azure storage accounts.
#HUGGINGFACE_MODEL_DIR = (
#    Path("/mnt/my_input/huggingface_models/")
#    if RUN_ON_AML
#    else Path("huggingface_models/")
#)
HUGGINGFACE_MODEL_DIR = Path(os.environ.get("TRANSFORMERS_CACHE", "huggingface_models/") )
#TRAINED_MODEL_DIR = (
#    Path("/mnt/my_output/trained_models/") if RUN_ON_AML else Path("trained_models/")
#)
TRAINED_MODEL_DIR = Path(os.environ.get("CHECKPOINT_DIR", "trained_models") )
# LOG_DIR = Path("/mnt/my_output/logs/") if RUN_ON_AML else Path("/brtx/601-nvme1/estengel/calflow_calibration/benchclamp/logs/")
# TODO(Elias): change back once done debugging
LOG_DIR = Path("/mnt/my_output/logs/") if RUN_ON_AML else Path("/brtx/602-nvme1/estengel/calflow_calibration/benchclamp/logs/")
# LOG_DIR = Path("/mnt/my_output/logs/") if RUN_ON_AML else Path("/brtx/602-nvme1/estengel/ambiguous_parsing/benchclamp/logs/")
# LOG_DIR = Path("/mnt/my_output/logs/") if RUN_ON_AML else Path("/home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs")
VERSION = "1.0"

LRS: List[float] = [1e-4, 1e-5]
LRS_FOR_T5_XL: List[float] = [1e-4]
BEAM_SIZE = 5

# TRAIN_MAX_STEPS = 10000
TRAIN_MAX_STEPS = 30000
STEPS_PER_SAVE = 5000
SEARCH_MAX_STEPS = 1000
BATCH_SIZE_PER_DEVICE = 4
# Effective batch size = BATCH_SIZE_PER_DEVICE * GRAD_ACCUMULATION_STEPS = 32

# Most experiments were run with 32 GB GPUS for the larger models (t5-large and t5-xl).
# To use less memory GPUs, you might have to split the model into more GPUs
# We currently do not support distributed training / inference, so all GPUs need to reside on the same machine.
TRAIN_MODEL_CONFIGS: List[ClampModelConfig] = [
    T5ModelConfig(
        model_id="t5-small-lm-adapt",
        model_loc=HUGGINGFACE_MODEL_DIR / "t5-small-lm-adapt",
        device_map={0: list(range(4)), 1: list(range(4, 12))}
        if torch.cuda.device_count() >= 2
        else None,
    ),
    T5ModelConfig(
        model_id="t5-base-lm-adapt",
        model_loc=HUGGINGFACE_MODEL_DIR / "t5-base-lm-adapt",
        device_map={0: list(range(4)), 1: list(range(4, 12))}
        if torch.cuda.device_count() >= 2
        else None,
    ),
    T5ModelConfig(
        model_id="t5-large-lm-adapt",
        model_loc=HUGGINGFACE_MODEL_DIR / "t5-large-lm-adapt",
        device_map={
            0: list(range(3)),
            1: list(range(3, 10)),
            2: list(range(10, 17)),
            3: list(range(17, 24)),
        }
        if torch.cuda.device_count() >= 4
        else {0: list(range(10)), 1: list(range(10, 24))}
        if torch.cuda.device_count() >= 2
        else None,
    ),
    T5ModelConfig(
        model_id="t5-xl-lm-adapt",
        model_loc=HUGGINGFACE_MODEL_DIR / "t5-xl-lm-adapt",
        device_map={
            0: list(range(3)),
            1: list(range(3, 10)),
            2: list(range(10, 17)),
            3: list(range(17, 24)),
        }
        if torch.cuda.device_count() >= 4
        else {0: list(range(10)), 1: list(range(10, 24))}
        if torch.cuda.device_count() >= 2
        else None,
    ),
    CodeT5ModelConfig(
        model_id="codet5-base",
        model_loc=HUGGINGFACE_MODEL_DIR / "codet5-base",
        device_map={0: list(range(4)), 1: list(range(4, 12))}
        if torch.cuda.device_count() >= 2
        else None,
    ),
    BartModelConfig(
        model_id="bart-large", model_loc=HUGGINGFACE_MODEL_DIR / "bart-large",
        device_map={
            0: list(range(3)),
            1: list(range(3, 10)),
            2: list(range(10, 17)),
            3: list(range(17, 24)),
        }
        if torch.cuda.device_count() >= 4
        else {0: list(range(10)), 1: list(range(10, 24))}
        if torch.cuda.device_count() >= 2
        else None,
    ),
    BartModelConfig(
        model_id="bart-base", model_loc=HUGGINGFACE_MODEL_DIR / "bart-base",
        # device_map={0: list(range(4)), 1: list(range(4, 12))}
        # if torch.cuda.device_count() >= 2
        # else None,
    ),
]

BATCH_SIZE_PER_DEVICE_OVERRIDES: Dict[str, int] = {
    f"{lm}_{dataset}_{inp}_{split_id}_{lr}": batch_size
    for lm in ["t5-xl-lm-adapt", "t5-large-lm-adapt"]
    for dataset in ["spider", "cosql", "calflow", "tree_dst", "lamp"]
    for inp, batch_size in [
        ("past_none_db_val", 1),
        ("past_one_db_val", 1),
        ("past_all_db_val", 1),
        ("last_agent", 2),
        ("last_user", 2),
    ]
    for lr in ["0.0001", "1e-5"]
    for split_id in ["low_0", "low_1", "low_2", "medium_0", "all", "tiny"]
}
BATCH_SIZE_PER_DEVICE_OVERRIDES.update(
    {
        f"{lm}_{dataset}_{inp}_{split_id}_{lr}": batch_size
        for lm in ["t5-base-lm-adapt", "bart-base"]
        for dataset in ["spider"]
        for inp, batch_size in [
            ("past_none_db_val", 3), 
            ("past_one_db_val", 3),
            ("past_all_db_val", 3),
        ]
        for lr in ["0.0001", "1e-5"]
        for split_id in ["low_0", "low_1", "low_2", "medium_0", "all", "tiny"]
    }
)
BATCH_SIZE_PER_DEVICE_OVERRIDES.update(
    {
        f"{lm}_{dataset}_{inp}_{split_id}_{lr}": batch_size
        for lm in ["t5-base-lm-adapt", "bart-base"]
        for dataset in ["cosql"]
        for inp, batch_size in [
            ("past_none_db_val", 3),
            ("past_one_db_val", 3),
            ("past_all_db_val", 2),
        ]
        for lr in ["0.0001", "1e-5"]
        for split_id in ["low_0", "low_1", "low_2", "medium_0", "all", "tiny"]
    }
)


def create_train_exp(
    exp_name: str,
    model_config: ClampModelConfig,
    data_config: ClampDataConfig,
    learning_rate: float,
) -> TrainExperiment:
    model, tokenizer, seq2seq_settings = model_config.setup_model()
    data_config.tokenizer = tokenizer
    train_data, dev_data, _ = data_config.setup_data()
    is_encoder_decoder = not isinstance(model_config, GPT2ModelConfig)

    if isinstance(tokenizer, T5ClampTokenizer):
        output_sequences = []
        for datum in train_data:
            if seq2seq_settings.output_surround.starts_with_space:
                output_sequences.append(" " + datum.canonical)
            else:
                output_sequences.append(datum.canonical)
        tokenizer.update_tokenizer_with_output_sequences(output_sequences)

    per_device_batch_size = BATCH_SIZE_PER_DEVICE_OVERRIDES.get(
        exp_name, BATCH_SIZE_PER_DEVICE
    )
    grad_acc_steps = 32 // per_device_batch_size

    return TrainExperiment(
        train_data=train_data,
        eval_data=dev_data,
        model=model,
        tokenizer=tokenizer,
        is_encoder_decoder=is_encoder_decoder,
        seq2seq_settings=seq2seq_settings,
        training_args=TrainingArguments(
            output_dir=f"{TRAINED_MODEL_DIR}/{VERSION}/{exp_name}/",
            learning_rate=learning_rate,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            evaluation_strategy=IntervalStrategy.STEPS,
            max_steps=TRAIN_MAX_STEPS,
            save_steps=STEPS_PER_SAVE,
            warmup_steps=TRAIN_MAX_STEPS // 10,
            group_by_length=True,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            gradient_accumulation_steps=grad_acc_steps,
            adafactor=True,
            logging_dir=LOG_DIR / VERSION,
        ),
        log_dir=LOG_DIR / VERSION,
    )


def create_eval_exp(
    exp_name: str,
    model_config: ClampModelConfig,
    data_config: ClampDataConfig,
    problem_type: Literal[ "unconstrained-beam", "unconstrained-greedy", "constrained"],
    is_dev: bool,
) -> Experiment:
    model, tokenizer, _ = model_config.setup_model()
    data_config.tokenizer = tokenizer
    train_data, dev_data, test_data = data_config.setup_data()
    is_encoder_decoder = not isinstance(model_config, GPT2ModelConfig)

    lm: Seq2SeqModel
    if is_encoder_decoder:
        lm = Seq2SeqBart(
            pretrained_model_dir=str(model_config.model_loc),
            model=model,
            clamp_tokenizer=tokenizer,
        )
    else:
        lm = Seq2SeqGPT2(
            pretrained_model_dir=str(model_config.model_loc),
            model=model,
            clamp_tokenizer=tokenizer,
        )

    if problem_type == "constrained":
        constrained = True
        beam_size = BEAM_SIZE
    elif problem_type == "unconstrained-beam":
        constrained = False
        beam_size = BEAM_SIZE
    elif problem_type == "unconstrained-greedy":
        constrained = False
        beam_size = 1
    else:
        raise ValueError(f"{problem_type} not allowed")

    eval_data = dev_data if is_dev else test_data

    if isinstance(data_config, BenchClampDatasetConfig):
        if data_config.dataset_name == BenchClampDataset.Overnight.value:
            # Overnight has a different grammar strategy so handling separately
            pieces = OvernightPieces.from_dir(
                lm.tokenizer,
                OVERNIGHT_DATA_DIR_AZURE,
                data_config.domain,  # type: ignore
                is_dev=is_dev,
                k=BEAM_SIZE,
                output_type=OutputType.MeaningRepresentation,
                simplify_logical_forms=True,
                # TODO: Set prefix_with_space properly by inspecting `lm`
                prefix_with_space=True,
            )
            max_steps = (
                max(
                    len(lm.tokenizer.tokenize(" " + canon))
                    for canon in pieces.denotation_metric.canonical_to_denotation
                )
                + 3  # +3 to be safe
            )

            partial_parse_builder: Callable[[Datum], PartialParse]
            if constrained:
                partial_parse_builder = pieces.partial_parse_builder  # type: ignore
            else:
                partial_parse = StartsWithSpacePartialParse(lm.tokenizer)
                partial_parse_builder = lambda _: partial_parse

            parser = make_semantic_parser(
                train_data,
                lm,  # type: ignore
                False,
                max_steps,
                beam_size,
                partial_parse_builder,
                lambda _datum: max_steps,
            )

            return Experiment(
                model=parser,
                client=lm,
                metrics={
                    "exact_match": TopKExactMatch(beam_size),
                    "denotation": pieces.denotation_metric,
                },
                test_data=test_data,
                log_dir=LOG_DIR / VERSION,
            )

        else:
            # Everything other than Overnight in BenchClamp
            train_length_pairs = []
            for datum in train_data:
                num_input_tokens = len(tokenizer.tokenize(datum.natural))
                num_output_tokens = len(tokenizer.tokenize(datum.canonical)) + 1
                train_length_pairs.append((num_input_tokens, num_output_tokens))

            print("Computing max steps regression model parameters ...")
            max_steps_intercept, max_steps_slope = compute_and_print_fit(
                train_length_pairs, 10, 1
            )
            print("Done")
            partial_parse_builder = create_partial_parse_builder(
                constrained, data_config, tokenizer
            )
            max_steps_fn = lambda _datum: min(
                int(
                    len(tokenizer.tokenize(_datum.natural)) * max_steps_slope
                    + max_steps_intercept
                ),
                1000,
            )


            parser = make_semantic_parser(
                train_data=train_data,  # type: ignore
                lm=lm,  # type: ignore
                use_gpt3=False,
                global_max_steps=SEARCH_MAX_STEPS,
                beam_size=beam_size,
                partial_parse_builder=partial_parse_builder,
                max_steps_fn=max_steps_fn,
            )
            metrics: Dict[str, Metric[Sequence[str], FullDatum]] = {
                "exact_match": TopKExactMatch(beam_size)
            }
            if data_config.dataset_name in [
                BenchClampDataset.CalFlowV2.value,
                BenchClampDataset.TreeDST.value,
            ]:
                metrics["lispress_match"] = TopKLispressMatch(beam_size)
            elif data_config.dataset_name in [
                BenchClampDataset.Spider.value,
                BenchClampDataset.CoSQL.value,
            ] and problem_type in ["constrained", "unconstrained-beam"]:
                metrics["test_suite_execution_acc"] = SQLTestSuiteMatch(
                    db_path=str(TEST_SUITE_DATABASE_PATH),
                    test_suite_path=str(TEST_SUITE_PATH),
                    table_file=str(SPIDER_TABLES_FILE)
                    if data_config.dataset_name == BenchClampDataset.Spider.value
                    else str(COSQL_TABLES_FILE),
                    log_dir=str(LOG_DIR / VERSION / exp_name),
                )

            print("Returned experiment")
            return Experiment(
                model=parser,
                metrics=metrics,
                test_data=eval_data,
                client=lm,
                log_dir=LOG_DIR / VERSION,
            )

    raise ValueError("Could not create eval experiment with inputs")


def create_exps_dict() -> Tuple[
    Dict[str, Callable[[], TrainExperiment]], Dict[str, Callable[[], Experiment]]
]:
    train_exps_dict: Dict[str, Callable[[], TrainExperiment]] = {}
    eval_exps_dict: Dict[str, Callable[[], Experiment]] = {}
    for data_config in BENCHCLAMP_DATA_CONFIGS:
        for train_model_config in TRAIN_MODEL_CONFIGS:
            # List of checkpoints to evaluate (names and paths)
            trained_model_locs: List[Tuple[str, Path]] = []
            learning_rates = (
                LRS_FOR_T5_XL
                if train_model_config.model_id.startswith("t5-xl")
                else LRS
            )
            for learning_rate in learning_rates:
                train_exp_name = f"{train_model_config.model_id}_{data_config.data_id}_{learning_rate}"
                train_exps_dict[train_exp_name] = functools.partial(
                    create_train_exp,
                    train_exp_name,
                    train_model_config,
                    data_config,
                    learning_rate,
                )
                trained_model_locs.extend(
                    [
                        (
                            f"{train_exp_name}_{epoch}",
                            TRAINED_MODEL_DIR
                            / VERSION
                            / train_exp_name
                            / f"checkpoint-{epoch}",
                        )
                        for epoch in list(
                            range(STEPS_PER_SAVE, TRAIN_MAX_STEPS + 1, STEPS_PER_SAVE)
                        )
                    ]
                )

            dev_results: Dict[Tuple[str, Path], float] = {}
            dev_complete = False
            for trained_model_id, trained_model_loc in trained_model_locs:
                eval_model_config = dataclasses.replace(
                    train_model_config,
                    model_id=trained_model_id,
                    model_loc=trained_model_loc,
                )
                eval_exp_name = f"{trained_model_id}_dev_eval"
                # We assume sharding is not necessary for dev set TODO: Relax this assumption
                results_file_path = LOG_DIR / VERSION / eval_exp_name / "results.json"
                # pdb.set_trace()
                if Path.exists(results_file_path):
                    dev_complete = True
                    dev_results[(trained_model_id, trained_model_loc)] = json.load(
                        open(results_file_path)
                    )["exact_match/top1"]
                else:
                    # dev_complete = False
                    eval_exps_dict[eval_exp_name] = functools.partial(
                        create_eval_exp,
                        eval_exp_name,
                        eval_model_config,
                        data_config,
                        "unconstrained-greedy",
                        is_dev=True,
                    )
 
            if dev_complete and len(dev_results) > 0:
                print(f"All dev expts complete. Results gathered.\n{dev_results}")
                best_trained_model_info = max(
                    # pylint: disable=cell-var-from-loop
                    dev_results,
                    key=lambda key: dev_results[key],
                )

                if best_trained_model_info is not None:
                    best_model_id, best_model_loc = best_trained_model_info
                    eval_model_config = dataclasses.replace(
                        train_model_config,
                        model_id=train_model_config.model_id,
                        model_loc=best_model_loc,
                    )
                    for constrained in ["constrained", "unconstrained-beam"]:
                    # for constrained in ["unconstrained-beam"]:
                        eval_exp_name = (
                            f"{best_model_id}_test_eval_{constrained}_bs_{BEAM_SIZE}"
                        )
                        eval_exps_dict[eval_exp_name] = functools.partial(
                            create_eval_exp,
                            eval_exp_name,
                            eval_model_config,
                            data_config,
                            constrained,  # type: ignore
                            is_dev=False,
                        )
                        # TODO (elias): May need to delete this if it's causing 
                        # model to hang up because of missing dev experiments
                        dev_eval_exp_name = (
                            f"{best_model_id}_dev_eval_{constrained}_bs_{BEAM_SIZE}"
                        )
                        eval_exps_dict[dev_eval_exp_name] = functools.partial(
                            create_eval_exp,
                            dev_eval_exp_name,
                            eval_model_config,
                            data_config,
                            constrained,  # type: ignore
                            is_dev=True,
                        )
    return train_exps_dict, eval_exps_dict


def build_config(
    log_dir,  # pylint: disable=unused-argument
    **kwargs: Any,  # pylint: disable=unused-argument
) -> Dict[str, Callable[[], Union[TrainExperiment, Experiment]]]:
    sys.setrecursionlimit(50000)
    expts: Dict[str, Callable[[], Union[TrainExperiment, Experiment]]] = {}
    train_expts, eval_expts = create_exps_dict()
    expts.update(train_expts)
    expts.update(eval_expts)
    return expts
