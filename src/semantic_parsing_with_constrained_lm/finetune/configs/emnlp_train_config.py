# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" This config file is for running training experiments for the EMNLP camera ready. """

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from transformers import IntervalStrategy, TrainingArguments
from typing_extensions import Literal

from semantic_parsing_with_constrained_lm.datum import FullDatum
from semantic_parsing_with_constrained_lm.domains.calflow import (
    CalflowOutputLanguage,
    read_calflow_jsonl,
)
from semantic_parsing_with_constrained_lm.domains.overnight import OutputType, OvernightPieces
from semantic_parsing_with_constrained_lm.domains.qdmr_break import (
    BreakDataType,
    BreakPieces,
    BreakSamplingType,
)
from semantic_parsing_with_constrained_lm.lm import TRAINED_MODEL_DIR
from semantic_parsing_with_constrained_lm.paths import DOMAINS_DIR
from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer
from semantic_parsing_with_constrained_lm.train_model_setup import BartModelConfig
from semantic_parsing_with_constrained_lm.finetune.lm_finetune import PRETRAINED_MODEL_DIR, TrainExperiment


def read_data(
    dataset: str,  # one of "calflow", "overnight", "break"
    output_type: Union[CalflowOutputLanguage, OutputType, BreakDataType],
    overnight_domain: Optional[str] = None,
    tokenizer: Optional[ClampTokenizer] = None,
) -> Tuple[Sequence[FullDatum], Sequence[FullDatum]]:
    print(f"Read data {dataset} {output_type}")
    if dataset == "calflow" and output_type in [
        CalflowOutputLanguage.Canonical,
        CalflowOutputLanguage.Lispress,
    ]:
        calflow_train_data_file = (
            DOMAINS_DIR / "calflow/data/train_300_stratified.jsonl"
        )
        calflow_eval_data_file = DOMAINS_DIR / "calflow/data/dev_100_uniform.jsonl"
        train_data = [
            datum
            for datum in read_calflow_jsonl(calflow_train_data_file, output_type)
            if datum.canonical is not None
        ]
        eval_data = [
            datum
            for datum in read_calflow_jsonl(calflow_eval_data_file, output_type)[:100]
            if datum.canonical is not None
        ]
        return train_data, eval_data

    elif (
        dataset == "overnight"
        and output_type in [OutputType.Utterance, OutputType.MeaningRepresentation]
        and overnight_domain is not None
        and tokenizer is not None
    ):
        overnight_pieces = OvernightPieces.from_dir(
            tokenizer,
            DOMAINS_DIR / "overnight/data",
            overnight_domain,
            is_dev=True,
            k=1,
            output_type=output_type,
            simplify_logical_forms=True,
            prefix_with_space=True,
        )
        return overnight_pieces.train_data[:200], overnight_pieces.test_data[:100]

    elif (
        dataset == "break"
        and output_type in [BreakDataType.nested, BreakDataType.qdmr]
        and tokenizer is not None
    ):
        break_pieces = BreakPieces.build(
            tokenizer,
            data_type=output_type,
            train_sampling_type=BreakSamplingType.proportional,
            test_sampling_type=BreakSamplingType.random,
            train_total=200,
            test_total=100,
            seed=0,
        )
        return break_pieces.train_data, break_pieces.test_data

    else:
        raise ValueError("read_data() called with incompatible arguments.")


def create_exp(
    exp_name: str,
    dataset: Literal["calflow", "overnight", "break"],
    output_type: Union[CalflowOutputLanguage, OutputType, BreakDataType],
    model_type: str,
    overnight_domain: Optional[str] = None,
) -> TrainExperiment:
    model_config = BartModelConfig(
        model_id="Bart", model_loc=Path(PRETRAINED_MODEL_DIR)
    )
    model, tokenizer, seq2seq_settings = model_config.setup_model()
    train_data, eval_data = read_data(
        dataset, output_type, overnight_domain=overnight_domain, tokenizer=tokenizer
    )
    return TrainExperiment(
        train_data=train_data,
        eval_data=eval_data,
        model=model,
        tokenizer=tokenizer,
        is_encoder_decoder=model_type == "Bart",
        seq2seq_settings=seq2seq_settings,
        training_args=TrainingArguments(
            output_dir=f"{TRAINED_MODEL_DIR}/{exp_name}/",
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            evaluation_strategy=IntervalStrategy.STEPS,
            learning_rate=1e-5,
            max_steps=10000,
            save_steps=10000,
            warmup_steps=1000,
            group_by_length=True,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
        ),
        log_dir=Path(f"logs/{exp_name}/"),
    )


def add_exp_to_dict(
    exp_name: str,
    exps_dict: Dict[str, Callable[[], TrainExperiment]],
    dataset: Literal["calflow", "overnight", "break"],
    output_type: Union[CalflowOutputLanguage, OutputType, BreakDataType],
    model_type: str,
    overnight_domain: Optional[str] = None,
):
    exps_dict[exp_name] = lambda: create_exp(
        exp_name,
        dataset,
        output_type,
        model_type=model_type,
        overnight_domain=overnight_domain,
    )


def build_config(
    log_dir,  # pylint: disable=unused-argument
    **kwargs: Any,  # pylint: disable=unused-argument
) -> Dict[str, Callable[[], TrainExperiment]]:
    MODEL_NAME = "Bart"
    OVERNIGHT_DOMAINS = (
        "housing",
        "calendar",
        "socialnetwork",
        "basketball",
        "blocks",
        "publications",
        "recipes",
        "restaurants",
    )
    result: Dict[str, Callable[[], TrainExperiment]] = {}
    for output_type in CalflowOutputLanguage:
        exp_name = f"calflow_{output_type}"
        add_exp_to_dict(
            exp_name,
            result,
            dataset="calflow",
            output_type=output_type,
            model_type=MODEL_NAME,
            overnight_domain=None,
        )
    for domain in OVERNIGHT_DOMAINS:
        for output_type in OutputType:
            exp_name = f"overnight_{domain}_{output_type}"
            add_exp_to_dict(
                exp_name,
                result,
                dataset="overnight",
                output_type=output_type,
                model_type=MODEL_NAME,
                overnight_domain=domain,
            )
    for output_type in BreakDataType:
        exp_name = f"break_{output_type}"
        add_exp_to_dict(
            exp_name,
            result,
            dataset="break",
            output_type=output_type,
            model_type=MODEL_NAME,
        )

    return result
