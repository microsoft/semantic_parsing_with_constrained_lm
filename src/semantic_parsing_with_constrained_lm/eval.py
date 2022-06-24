# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import dataclasses
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generic, List, Optional, Sequence, TextIO, TypeVar

from semantic_parsing_with_constrained_lm.datum import FullDatum, FullDatumSub
from semantic_parsing_with_constrained_lm.model import ModelResult

Pred = TypeVar("Pred")
Target = TypeVar("Target")


# TODO: Replcae this with a more flexible function suited to each domain
def exact_match_with_logging(
    test_datum: FullDatum, kbest: Sequence[ModelResult]
) -> bool:
    gold = (
        test_datum.canonical.strip(" ")
        if test_datum.canonical is not None
        else "UNREACHABLE"
    )
    pred = kbest[0].text.strip(" ") if kbest else ""
    print()
    print(f"context:   {test_datum.agent_context}")
    print(f"natural:   {test_datum.natural}")
    print(f"predicted: {pred}")
    print(f"gold:      {gold}")
    result = gold == pred
    print(f"is correct: {result}")
    beam_result = False
    for i, pred_i in enumerate(kbest):
        stripped = pred_i.text.strip(" ")
        beam_result = beam_result or gold == stripped
        print(f"Beam {i} [{pred_i.cost:.3f}]: {stripped}")
        print(f"is correct@{i}: {beam_result}")
    print()
    return result


class Metric(Generic[Pred, Target], ABC):
    """Used to measure goodness of model results compared to the ground truth.

    Stateful over the duration of an experiment run."""

    @abstractmethod
    def update(self, pred: Pred, target: Target) -> Dict[str, Optional[str]]:
        """Uses `target` and the model predictions `pred` to update the state."""
        pass

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """Uses the state to compute the final results."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reinitializes the state."""
        pass


@dataclass
class TopKExactMatch(Metric[Sequence[str], FullDatumSub]):
    k: int
    correct: List[int] = dataclasses.field(init=False)
    total: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.reset()

    def _is_correct(self, pred: str, target: FullDatumSub) -> bool:
        """Can be overridden by child classes."""
        return pred == target.canonical

    def update(
        self, preds: Sequence[str], target: FullDatumSub
    ) -> Dict[str, Optional[str]]:
        self.total += 1
        found_correct = False
        result: Dict[str, Optional[str]] = {}
        for i, pred in enumerate(preds[: self.k]):
            correct = self._is_correct(pred, target)
            found_correct |= correct
            self.correct[i] += found_correct
            result[f"rank{i + 1}"] = "correct" if correct else "incorrect"
            result[f"top{i + 1}"] = "correct" if found_correct else "incorrect"

        # Handle when we have fewer predictions than self.k
        for i in range(len(preds), self.k):
            self.correct[i] += found_correct
            result[f"rank{i + 1}"] = "incorrect"
            result[f"top{i + 1}"] = "correct" if found_correct else "incorrect"

        return result

    def compute(self) -> Dict[str, float]:
        result = {}
        for i in range(self.k):
            result[f"top{i + 1}"] = self.correct[i] / self.total
        return result

    def reset(self) -> None:
        self.correct = [0] * self.k
        self.total = 0


class Logger(Generic[Pred, Target], AbstractContextManager):
    """Experiment logger interface for capturing model outputs for a
    given target and logging them in some form. Useful for things like
    producing error tables.

    The logger implements context manager and must be wrapped in a
    `with` statement to be used.
    """

    @abstractmethod
    def log(self, predictions: Pred, target: Target, metrics: Dict[str, Optional[str]]):
        pass


class ExactMatchErrorLogger(Logger[Sequence[ModelResult], FullDatumSub]):
    """A logger that logs in 2 files:
    - base_path / errors.csv - CSV with model error using exact match metric.
    - base_path / correct.scv - CSV with correct model predictions.
    """

    def __init__(self, base_path: Path):
        self._base_path = base_path
        self._err_file: Optional[TextIO] = None
        self._err_writer = None
        self._corr_file: Optional[TextIO] = None
        self._corr_writer = None

    def __enter__(self):
        assert not self._err_file and not self._corr_file
        self._err_file = open(self._base_path / "errors.csv", "w")
        self._err_writer = csv.writer(
            self._err_file, delimiter=",", quoting=csv.QUOTE_MINIMAL
        )
        self._corr_file = open(self._base_path / "correct.csv", "w")
        self._corr_writer = csv.writer(
            self._corr_file, delimiter=",", quoting=csv.QUOTE_MINIMAL
        )

        self._err_writer.writerow(["Natural", "Gold", "Predicted"])  # type: ignore
        self._corr_writer.writerow(["Natural", "Gold"])  # type: ignore

    def __exit__(self, *_):
        assert self._err_file and self._corr_file
        self._err_file.close()
        self._corr_file.close()

    def log(
        self,
        predictions: Sequence[ModelResult],
        target: FullDatumSub,
        metrics: Dict[str, Optional[str]],
    ):
        pred = predictions[0].text if len(predictions) else ""
        if pred != target.canonical:
            self._err_writer.writerow([target.natural, target.canonical, pred])  # type: ignore
            self._err_file.flush()  # type: ignore
        else:
            self._corr_writer.writerow([target.natural, target.canonical])  # type: ignore
            self._corr_file.flush()  # type: ignore
