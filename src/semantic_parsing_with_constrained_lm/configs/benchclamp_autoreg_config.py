# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Config to run evaluation experiments with BenchCLAMP with GPT-3 based language models with a few shot prompted
approach.
"""
import pdb 
import functools
import itertools
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Tuple, Union, List
import torch 

from typing_extensions import Literal

from semantic_parsing_with_constrained_lm.configs.lib.benchclamp import create_partial_parse_builder
from semantic_parsing_with_constrained_lm.configs.lib.common import (
    BM25Indexer,
    PromptOrder,
    make_semantic_parser,
)

from semantic_parsing_with_constrained_lm.domains.sql.sql_metric import SQLTestSuiteMatch
from semantic_parsing_with_constrained_lm.configs.lib.benchclamp import (
    COSQL_TABLES_FILE,
    SPIDER_TABLES_FILE,
    TEST_SUITE_DATABASE_PATH,
    TEST_SUITE_PATH,
    create_partial_parse_builder,
)

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
from semantic_parsing_with_constrained_lm.eval import Metric, TopKExactMatch
from semantic_parsing_with_constrained_lm.fit_max_steps import compute_and_print_fit
from semantic_parsing_with_constrained_lm.lm_gpt2 import Seq2SeqGPT2, IncrementalGPT2
from semantic_parsing_with_constrained_lm.paths import OVERNIGHT_DATA_DIR_AZURE
from semantic_parsing_with_constrained_lm.run_exp import Experiment
from semantic_parsing_with_constrained_lm.finetune.lm_finetune import TrainExperiment

from semantic_parsing_with_constrained_lm.train_model_setup import (
    BartModelConfig,
    ClampModelConfig,
    CodeGenModelConfig,
    CodeT5ModelConfig,
    GPT2ModelConfig,
    T5ModelConfig,
)
from semantic_parsing_with_constrained_lm.configs.benchclamp_config import HUGGINGFACE_MODEL_DIR

# LOG_DIR = Path("logs/")
# VERSION = "1.10"

# TODO(Elias): change back once done debugging
LOG_DIR = Path("/brtx/604-nvme1/estengel/logit_pr/logs/")
VERSION = "1.0"

BEAM_SIZE = 5
# SEARCH_MAX_STEPS = 500
SEARCH_MAX_STEPS = 50

EVAL_MODEL_CONFIGS: List[ClampModelConfig] = [
    CodeGenModelConfig(
        model_id="codegen-350M",
        model_loc=HUGGINGFACE_MODEL_DIR / "codegen-350M",
        device_map={0: list(range(11)), 1: list(range(11, 20))}
        if torch.cuda.device_count() >= 2
        else None,
    ),
    CodeGenModelConfig(
        model_id="codegen-2B",
        model_loc=HUGGINGFACE_MODEL_DIR / "codegen-2B",
        device_map={0: list(range(15)), 1: list(range(15, 32))}
        if torch.cuda.device_count() >= 2
        else {0: list(range(8)),
        1: list(range(8,16)),
        2: list(range(16,24)),
        3: list(range(24,32))}
        if torch.cuda.device_count() == 4
        else {0: list(range(4)),
        1: list(range(4,8)),
        2: list(range(8,12)),
        3: list(range(12,16)),
        4: list(range(16,20)),
        5: list(range(20,24)),
        6: list(range(24,28)),
        7: list(range(28,32))}
        if torch.cuda.device_count() == 8
        else None,
    ),
    CodeGenModelConfig(
        model_id="codegen-6B",
        model_loc=HUGGINGFACE_MODEL_DIR / "codegen-6B",
        device_map={0: list(range(11)), 1: list(range(11, 22)), 2: list(range(22, 33))}
        if torch.cuda.device_count() >= 3
        else None,
    ),
    CodeGenModelConfig(
        model_id="codegen-16B",
        model_loc=HUGGINGFACE_MODEL_DIR / "codegen-16B",
        device_map={0: list(range(9)), 1: list(range(9, 17)), 2: list(range(17, 26)), 3: list(range(26, 34))}
        if torch.cuda.device_count() == 4
        else None,
    ),
    ]


def create_eval_exp(
    exp_name: str,
    model_config: ClampModelConfig,
    data_config: ClampDataConfig,
    problem_type: Literal["constrained", "unconstrained-beam", "unconstrained-greedy"],
    is_dev: bool,
    prompt_order: PromptOrder,
) -> Experiment:

    train_data, dev_data, test_data = data_config.setup_data()
    # lm = IncrementalOpenAIGPT3(engine=open_ai_model_name)

    model, tokenizer, _ = model_config.setup_model()
    data_config.tokenizer = tokenizer
    train_data, dev_data, test_data = data_config.setup_data()


    lm = IncrementalGPT2(
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
                similarity_method=BM25Indexer(),
                prompt_order=prompt_order,
                num_examples_per_prompt=5,
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
    for (
        data_config,
        open_ai_model,
        is_dev,
        constrained,
        prompt_order,
    ) in itertools.product(
        BENCHCLAMP_DATA_CONFIGS,
        ("codegen-350M", "codegen-2B", "codegen-6B", "codegen-16B"),
        (True, False),
        ("constrained", "unconstrained-beam", "unconstrained-greedy"),
        PromptOrder,
    ):
        dev_or_test = "dev" if is_dev else "test"
        eval_exp_name = (
            f"{open_ai_model}_{data_config.data_id}_{prompt_order.value}_{dev_or_test}_"
            f"eval_{constrained}_bs_{BEAM_SIZE}"
        )

        # get the right config 
        eval_model_config = [x for x in EVAL_MODEL_CONFIGS if x.model_id == open_ai_model][0]
        eval_exps_dict[eval_exp_name] = functools.partial(
            create_eval_exp,
            open_ai_model,
            eval_model_config,
            data_config,
            constrained,  # type: ignore
            is_dev=is_dev,
            prompt_order=prompt_order,
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
