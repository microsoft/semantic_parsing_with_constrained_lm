# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import bdb
import datetime
import importlib
import json
import pathlib
import re
import sys
import time
import traceback
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    AsyncContextManager,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import jsons
import torch
import typer

import semantic_parsing_with_constrained_lm
from semantic_parsing_with_constrained_lm.util import logger
from semantic_parsing_with_constrained_lm.async_tools import limits
from semantic_parsing_with_constrained_lm.datum import FullDatumSub
from semantic_parsing_with_constrained_lm.eval import Logger, Metric, exact_match_with_logging
from semantic_parsing_with_constrained_lm.lm import ClientType
from semantic_parsing_with_constrained_lm.model import Model, ModelResult
from semantic_parsing_with_constrained_lm.result import DatumResult
from semantic_parsing_with_constrained_lm.train_model_setup import TrainedModelNotFoundError

E = TypeVar("E")


@dataclass
class Experiment(Generic[FullDatumSub]):
    model: Model
    client: AsyncContextManager
    test_data: List[FullDatumSub]
    metrics: Mapping[str, Metric[Sequence[str], FullDatumSub]]
    log_dir: Optional[Path] = None
    loggers: Optional[List[Logger[Sequence[ModelResult], FullDatumSub]]] = None


class EvalSplit(str, Enum):
    """Controls which data is used for evaluation."""

    # 100-200 examples from the dev set.
    DevSubset = "dev-subset"
    # All dev set examples.
    DevFull = "dev-full"
    # 100-200 examples from the test set.
    TestSubset = "test-subset"
    # All the test set examples.
    TestFull = "test-full"
    # 100-200 examples from the training set.
    # Used as dev when we do not have access to test, and need to get results on full dev.
    TrainSubset = "train-subset"


def filter_exp_dict(
    # Using Iterable[Tuple[str, E]] is deprecated
    exps: Union[Iterable[Tuple[str, E]], Dict[str, Callable[[], E]]],
    exp_names: Optional[List[str]],
    exp_name_pattern: Optional[List[str]],
) -> Dict[str, Callable[[], E]]:
    if isinstance(exps, dict):
        exps_dict = exps
    else:
        exps_dict = {exp_name: (lambda exp=exp: exp) for exp_name, exp in exps}
    # Help out pyright
    exps_dict = cast(Dict[str, Callable[[], E]], exps_dict)

    if exp_names and exp_name_pattern:
        print("Cannot specify --exp-names and --exp-name-pattern together")
        return {}

    if exp_name_pattern:
        exp_names = [
            name
            for name in exps_dict.keys()
            if any(re.match(pat, name) for pat in exp_name_pattern)
        ]
        if not exp_names:
            print("--exp-name-pattern matched no experiment names")
            return {}
        print("Matched experiments:")
        for name in exp_names:
            print(name)
    elif not exp_names:
        exp_names = list(exps_dict.keys())

    error = False
    for exp_name in exp_names:
        if exp_name not in exps_dict:
            print(f"Experiment {exp_name} not found in config.")
            error = True
    if error:
        print("Names in config:")
        for name in exps_dict.keys():
            print(name)
        return {}

    filtered_exp_dict = {k: v for k, v in exps_dict.items() if k in exp_names}
    return filtered_exp_dict


async def run(
    exp_name: str,
    exp: Experiment,
    log_dir: Optional[pathlib.Path] = None,
    debug: bool = False,
    ids: Optional[List[str]] = None,
    rerun: bool = False,
    num_eval_examples: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    if log_dir is None:
        if exp.log_dir is None:
            print("At least one of log_dir and exp.log_dir needs to be provided")
            return
        log_dir = exp.log_dir

    loggers: List[Logger] = exp.loggers if exp.loggers else []

    if world_size == 1:
        exp_log_dir = log_dir / exp_name
    else:
        exp_log_dir = log_dir / f"{exp_name}_rank-{rank:02d}-of-{world_size:02d}"
    exp_log_dir.mkdir(exist_ok=True, parents=True)
    results_path = exp_log_dir / "results.json"
    if results_path.exists() and not rerun:
        print(f"Skipping {exp_name}, already finished")
        return
    print("********************")
    print(f"Running {exp_name} rank {rank} world size {world_size}")
    print("********************")
    now = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    all_metric_results: Dict[str, float] = {}

    test_data = (
        [datum for datum in exp.test_data if datum.dialogue_id in ids]
        if ids
        else exp.test_data
    )
    if not test_data:
        print(f"No test data! ids: {ids}")
        return

    print(f"Total test examples: {len(test_data)}")
    test_data = test_data[
        (rank * len(test_data))
        // world_size : ((rank + 1) * len(test_data))
        // world_size
    ]
    if num_eval_examples is not None:
        test_data = test_data[:num_eval_examples]

    print(f"Test examples this shard: {len(test_data)}")
    current_test_index = 0

    # Find past model outputs
    candidate_past_model_outputs: List[Tuple[pathlib.Path, List[Dict]]] = []
    for past_model_outputs_path in exp_log_dir.glob("model_outputs.*.jsonl"):
        candidate_past_model_outputs.append(
            (
                past_model_outputs_path,
                [json.loads(line) for line in open(past_model_outputs_path, "r")],
            )
        )
    if candidate_past_model_outputs:
        past_model_outputs_path, past_model_outputs_to_copy = max(
            candidate_past_model_outputs, key=lambda t: len(t[1])
        )
        if len(past_model_outputs_to_copy) > 0:
            print(
                f"*** Copying {len(past_model_outputs_to_copy)} past results from {past_model_outputs_path} ***"
            )
    else:
        past_model_outputs_to_copy = []

    with logger.intercept_output(
        exp_log_dir / f"stdout.{now}", exp_log_dir / f"stderr.{now}"
    ), open(
        exp_log_dir / f"model_outputs.{now}.jsonl", "w"
    ) as model_outputs_f, ExitStack() as logger_cm:
        for lg in loggers:
            logger_cm.enter_context(lg)

        try:
            for metric in exp.metrics.values():
                metric.reset()
            for test_datum, past_model_output in zip(
                test_data, past_model_outputs_to_copy
            ):
                current_test_index += 1
                assert test_datum.dialogue_id == past_model_output["test_datum_id"]
                assert (
                    test_datum.turn_part_index
                    == past_model_output["test_datum_turn_part_index"]
                )
                for metric in exp.metrics.values():
                    metric.update(past_model_output["outputs"], test_datum)
                model_outputs_f.write(json.dumps(past_model_output) + "\n")
            model_outputs_f.flush()

            start_time = time.time()
            first_unprocessed_test_index = current_test_index

            async with exp.client:
                async for kbest, test_datum in limits.map_async_limited(
                    exp.model.predict,
                    test_data[len(past_model_outputs_to_copy) :],
                    max_concurrency=1,
                    wrap_exception=not debug,
                ):
                    beam_search_text = [beam.text for beam in kbest]

                    all_metric_results_for_datum: Dict[str, Optional[str]] = {}
                    for metric_name, metric in exp.metrics.items():
                        metric_one_result = metric.update(beam_search_text, test_datum)
                        for key, value_str in metric_one_result.items():
                            all_metric_results_for_datum[
                                f"{metric_name}/{key}"
                            ] = value_str
                    print(
                        exp_log_dir, json.dumps(all_metric_results_for_datum, indent=4)
                    )
                    results = DatumResult(
                        test_datum.natural,
                        kbest,
                        beam_search_text,
                        all_metric_results_for_datum,
                        test_datum.dialogue_id,
                        test_datum.turn_part_index,
                        test_datum.agent_context,
                        test_datum.canonical,
                    )
                    model_outputs_f.write(jsons.dumps(results) + "\n")
                    model_outputs_f.flush()

                    for lg in loggers:
                        lg.log(kbest, test_datum, all_metric_results_for_datum)

                    # TODO: Delete this call and replace it with more flexible logging?
                    exact_match_with_logging(test_datum, kbest)
                    current_test_index += 1
                    print(f"Current test index: {current_test_index}")

            num_processed = current_test_index - first_unprocessed_test_index
            elapsed = time.time() - start_time
            per_item = elapsed / num_processed if num_processed else None
            print("Timing report:")
            print(f"- Items processed: {num_processed}")
            print(f"- Elapsed: {elapsed}")
            print(f"- Per item: {per_item}")
            with open(f"{exp_log_dir}/timings.json", "w") as f:
                json.dump(
                    {
                        "num_processed": num_processed,
                        "elapsed": elapsed,
                        "per_item": per_item,
                    },
                    f,
                )

            for metric_name, metric in exp.metrics.items():
                for key, value in metric.compute().items():
                    all_metric_results[f"{metric_name}/{key}"] = value

            print(results_path, json.dumps(all_metric_results, indent=4))

            if not ids:
                with open(results_path, "w") as results_f:
                    json.dump(all_metric_results, results_f)
        except (  # pylint: disable=try-except-raise
            KeyboardInterrupt,
            bdb.BdbQuit,
        ):
            # If we get Ctrl-C then we want to stop the entire program,
            # instead of just skipping this one experiment.
            raise
        except Exception as e:  # pylint: disable=broad-except
            if isinstance(e, limits.MapInnerException):
                if isinstance(e.__cause__, (KeyboardInterrupt, bdb.BdbQuit)):
                    # If we get Ctrl-C then we want to stop the entire program,
                    # instead of just skipping this one experiment.
                    raise e.__cause__

                # pylint: disable=no-member
                print(
                    f"Last test_datum: {e.orig_item} in experiment {exp_name}",
                    file=sys.stderr,
                )

            if debug:
                # If we're running inside a debugger, re-raise the
                # exception so that we can debug it.
                raise
            # Log the exception, and move onto the next item in `exps`.
            traceback.print_exc()


def main(
    config_name: str = typer.Option(...),
    log_dir: Optional[pathlib.Path] = typer.Option(None),
    debug: bool = typer.Option(False),
    exp_names: Optional[List[str]] = typer.Option(
        None
    ),  # pylint: disable=unused-argument
    exp_name_pattern: Optional[List[str]] = typer.Option(None),
    ids: Optional[List[str]] = typer.Option(None),
    rerun: bool = typer.Option(False),
    num_eval_examples: Optional[int] = typer.Option(None),
    model: ClientType = typer.Option(ClientType.GPT2),
    rank: int = typer.Option(0),
    world_size: int = typer.Option(1),
    results_dir: str = typer.Option("results"),
    eval_split: EvalSplit = typer.Option(EvalSplit.DevSubset),
):
    async def inner():
        nonlocal exp_names

        config_mod = importlib.import_module(config_name)
        kwargs = {
            "model": model,
            "results_dir": results_dir,
            "rank": rank,
            "eval_split": eval_split,
        }

        # TODO: Change log_dir argument into exp_log_dir
        exps = config_mod.build_config(log_dir, **kwargs)
        filtered_exp_dict = filter_exp_dict(exps, exp_names, exp_name_pattern)
        for exp_name in filtered_exp_dict:
            try:
                exp = filtered_exp_dict[exp_name]()
            except TrainedModelNotFoundError:
                # Trying to load models before training and saving them.
                continue

            if isinstance(exp, semantic_parsing_with_constrained_lm.run_exp.Experiment):  # type: ignore
                await run(
                    exp_name,
                    exp,
                    log_dir,
                    debug,
                    ids,
                    rerun,
                    num_eval_examples,
                    rank,
                    world_size,
                )
            else:
                del exp
                torch.cuda.empty_cache()

    with torch.no_grad():
        asyncio.run(inner())


if __name__ == "__main__":
    typer.run(main)
