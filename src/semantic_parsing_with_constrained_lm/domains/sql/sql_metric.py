# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from semantic_parsing_with_constrained_lm.datum import FullDatumSub
from semantic_parsing_with_constrained_lm.domains.sql.sql_datum import SqlDatum
from semantic_parsing_with_constrained_lm.eval import Metric


@dataclass
class SQLTestSuiteMatch(Metric[Sequence[str], FullDatumSub]):
    """
    Metric to evaluate SQL predictions. Uses the test-suite available here:
    https://github.com/taoyds/test-suite-sql-eval. To use this metric, clone this repo to a local
    directory and set `test_suite_path` to that directory.
    """

    db_path: str
    test_suite_path: str
    table_file: str
    log_dir: str
    schema_map: Dict[str, str] = dataclasses.field(init=False)
    predictions_map: Dict[Tuple[str, int], str] = dataclasses.field(init=False)
    gold_map: Dict[Tuple[str, int], str] = dataclasses.field(init=False)
    dialogue_to_turn_indices_map: Dict[str, List[int]] = dataclasses.field(init=False)

    def __post_init__(self):
        self.reset()

    def _is_correct(self, pred: str, target: SqlDatum) -> bool:
        """Can be overridden by child classes."""
        return pred == target.canonical

    def update(
        self, preds: Sequence[str], target: SqlDatum
    ) -> Dict[str, Optional[str]]:
        schema_name = target.schema_name
        self.schema_map[target.dialogue_id] = schema_name  # type: ignore
        self.predictions_map[(target.dialogue_id, target.turn_part_index)] = (  # type: ignore
            preds[0] if len(preds) > 0 else "dummy"
        )
        self.gold_map[(target.dialogue_id, target.turn_part_index)] = target.canonical  # type: ignore
        self.dialogue_to_turn_indices_map[target.dialogue_id].append(  # type: ignore
            target.turn_part_index  # type: ignore
        )
        return {}

    def compute(self, gold_file=None, pred_file=None) -> Dict[str, float]:
        # Run test suite using subprocess
        is_interaction = any(
            [
                len(turn_indices) > 1
                for _, turn_indices in self.dialogue_to_turn_indices_map.items()
            ]
        )
        if gold_file is None and pred_file is None:
            gold_file = self.log_dir + "/gold.txt"
            pred_file = self.log_dir + "/pred.txt"
            with open(gold_file, "w") as fp_gold, open(pred_file, "w") as fp_pred:
                for dialogue_id in self.dialogue_to_turn_indices_map:
                    for turn_index in self.dialogue_to_turn_indices_map[dialogue_id]:
                        gold = self.gold_map[(dialogue_id, turn_index)]
                        if gold.count(")") == 1 and gold.count("(") == 0:
                            gold = gold.replace(")", "")
                        if "faculty_participates_in" in gold:
                            gold = gold.replace(
                                "faculty_participates_in", "Faculty_participates_in"
                            )
                        fp_gold.write(
                            gold.replace(" . ", ".")
                            + "\t"
                            + self.schema_map[dialogue_id]
                            + "\n"
                        )
                        fp_pred.write(
                            self.predictions_map[(dialogue_id, turn_index)].replace(
                                " . ", "."
                            )
                            + "\n"
                        )

                    if is_interaction:
                        fp_gold.write("\n")
                        fp_pred.write("\n")

        process = subprocess.run(
            [
                "python3",
                "evaluation.py",
                "--gold",
                gold_file,
                "--pred",
                pred_file,
                "--db",
                self.db_path,
                "--table",
                self.table_file,
                "--etype",
                "all",
            ],
            cwd=self.test_suite_path,
            capture_output=True,
            text=True,
            check=True,
        )
        print("stdout:", process.stdout)
        print("stderr:", process.stderr)
        execution_acc = 0.0
        for line in process.stdout.split("\n"):
            if line.startswith("execution"):
                execution_acc = float(line.split()[5].strip())
                break
        result = {"execution_acc": execution_acc}
        print(result)
        return result

    def reset(self) -> None:
        self.predictions_map = {}
        self.gold_map = {}
        self.schema_map = {}
        self.dialogue_to_turn_indices_map = defaultdict(list)
