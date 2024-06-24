#!/bin/bash

valid_file="/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/TreeDST/test_all.jsonl"

for checkpoint in "/brtx/605-nvme1/estengel/calflow_calibration/benchclamp/text_to_treedst/1.0/t5-small-lm-adapt_tree_dst_last_user_all_0.0001/checkpoint-10000/" "/brtx/603-nvme1/estengel/calflow_calibration/benchclamp/text_to_treedst/1.0/t5-base-lm-adapt_tree_dst_last_user_all_0.0001/checkpoint-10000" "/brtx/603-nvme1/estengel/calflow_calibration/benchclamp/text_to_treedst/1.0/t5-large-lm-adapt_tree_dst_last_user_all_0.0001/checkpoint-10000/" "/brtx/604-nvme1/estengel/calflow_calibration/benchclamp/text_to_treedst/1.0/bart-base_tree_dst_last_user_all_0.0001/checkpoint-10000/" "/brtx/605-nvme1/estengel/calflow_calibration/benchclamp/text_to_treedst/1.0/bart-large_tree_dst_last_user_all_0.0001/checkpoint-10000/" 
do
    export CHECKPOINT_DIR=${checkpoint}
    export VALIDATION_FILE=${valid_file}
    sbatch slurm_scripts/text_to_lispress/treedst/get_logits.sh --export 
done
