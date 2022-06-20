# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json

import blobfile as bf

from semantic_parsing_with_constrained_lm.util.util import mean

low_splits = ["low_0", "low_1", "low_2"]
medium_and_high_splits = ["medium_0", "all"]


def read_finetuned_model_results():
    log_dir = "https://subhro.blob.core.windows.net/amulet/logs/1.10"
    model_dir = "https://subhro.blob.core.windows.net/amulet/trained_models/1.10"
    exp_names = []
    for model_id in [
        "bart-large",
        "codet5-base",
        "t5-base-lm-adapt",
        "t5-large-lm-adapt",
        "t5-xl-lm-adapt",
    ]:
        for dataset, context, splits in [
            ("calflow", "last_agent", low_splits),
            ("calflow", "last_user", medium_and_high_splits),
            ("tree_dst", "last_agent", low_splits),
            ("tree_dst", "last_user", medium_and_high_splits),
            ("mtop_en", "no_context", low_splits + medium_and_high_splits),
            ("overnight_blocks", "no_context", low_splits + medium_and_high_splits),
            ("spider", "past_none_db_val", low_splits + medium_and_high_splits),
            ("cosql", "past_one_db_val", low_splits),
            ("cosql", "past_all_db_val", medium_and_high_splits),
        ]:
            for split in splits:
                exp_name = "_".join([model_id, dataset, context, split])
                exp_names.append(exp_name)

    for exp_name in exp_names:
        print(exp_name)
        lrs = ["0.0001"] if exp_name.startswith("t5-xl") else ["0.0001", "1e-05"]
        for lr in lrs:
            for steps in [5000, 10000]:
                model_path = (
                    f"{model_dir}/{exp_name}_{lr}/checkpoint-{steps}/pytorch_model.bin"
                )
                results_path = (
                    f"{log_dir}/{exp_name}_{lr}_{steps}_dev_eval/results.json"
                )
                if not bf.exists(model_path):
                    print(f"Model Not Found for {exp_name} {lr} {steps}")
                elif not bf.exists(results_path):
                    print(f"Dev Not Found for {exp_name} {lr} {steps}")
        test_results_found = False
        for lr in lrs:
            for steps in [5000, 10000]:
                results_path = f"{log_dir}/{exp_name}_{lr}_{steps}_test_eval_constrained_bs_5_small_grammar/results.json"
                if bf.exists(results_path):
                    test_results_found = True
                    metric = "exact_match/top1"
                    if exp_name.startswith("spider") or exp_name.startswith("cosql"):
                        metric = "test_suite_execution_acc/execution_acc"
                    if exp_name.startswith("calflow") or exp_name.startswith(
                        "tree_dst"
                    ):
                        metric = "lispress_match/top1"
                    if exp_name.startswith("overnight"):
                        metric = "denotation/top1"
                    with bf.BlobFile(results_path) as fp:
                        results_dict = json.load(fp)
                        print(metric, results_dict[metric])

        if not test_results_found:
            print(f"Test Not Found for {exp_name}")
        print()


def read_gpt3_results():
    log_dir = "logs/1.10"
    exp_names = []
    for model_id in [
        "code-davinci-001",
        "text-davinci-001",
    ]:
        for dataset in ["calflow", "tree_dst", "mtop_en", "overnight_blocks"]:
            for split in low_splits + medium_and_high_splits:
                exp_name = "_".join([model_id, dataset, "no_context", split])
                exp_names.append(exp_name)

    for exp_name in exp_names:
        print(exp_name)
        results_path = f"{log_dir}/{exp_name}_2_test_eval_constrained_bs_5/results.json"
        if bf.exists(results_path):
            metric = "exact_match/top1"
            if exp_name.startswith("spider") or exp_name.startswith("cosql"):
                metric = "test_suite_execution_acc/execution_acc"
            if exp_name.startswith("calflow") or exp_name.startswith("tree_dst"):
                metric = "lispress_match/top1"
            if exp_name.startswith("overnight"):
                metric = "denotation/top1"
            with bf.BlobFile(results_path) as fp:
                results_dict = json.load(fp)
                print(metric, results_dict[metric])
        else:
            print(f"Test Not Found for {exp_name}")
        print()


def read_model_outputs(log_dir: str):
    candidate_model_outputs = []
    for past_model_outputs_path in bf.glob(log_dir + "/model_outputs.*.jsonl"):
        candidate_model_outputs.append(
            (
                past_model_outputs_path,
                [
                    json.loads(line)
                    for line in bf.BlobFile(past_model_outputs_path, "r")
                ],
            )
        )
    final_model_outputs_path, final_model_outputs = max(
        candidate_model_outputs, key=lambda t: len(t[1])
    )
    print(final_model_outputs_path, final_model_outputs[:2], "\n")


def read_test_all_results():
    log_dir = "https://subhro.blob.core.windows.net/amulet/logs/1.10"
    lrs = [1e-4]

    for model_id in ["t5-xl-lm-adapt"]:
        for exp_name in [
            "calflow_last_user_all",
            "tree_dst_last_user_all",
            "mtop_en_no_context_all",
        ]:
            print(model_id, exp_name)
            test_results_found = False
            for lr in lrs:
                for steps in [5000, 10000]:
                    scores = []
                    for rank in range(10):
                        results_path = (
                            f"{log_dir}/{model_id}_{exp_name}_{lr}_{steps}_test_eval_constrained_bs_5_"
                            f"rank-{rank:02d}-of-10/results.json"
                        )
                        metric = "exact_match/top1"
                        if exp_name.startswith("spider") or exp_name.startswith(
                            "cosql"
                        ):
                            metric = "test_suite_execution_acc/execution_acc"
                        if exp_name.startswith("calflow") or exp_name.startswith(
                            "tree_dst"
                        ):
                            metric = "lispress_match/top1"
                        if exp_name.startswith("overnight"):
                            metric = "denotation/top1"
                        with bf.BlobFile(results_path) as fp:
                            results_dict = json.load(fp)
                            print(metric, results_dict[metric])
                            scores.append(results_dict[metric])

                    if len(scores) > 0:
                        print("Avg:", mean(scores), len(scores))

            if not test_results_found:
                print(f"Test Not Found for {model_id} {exp_name}")
        print()


if __name__ == "__main__":
    read_finetuned_model_results()
    read_gpt3_results()
