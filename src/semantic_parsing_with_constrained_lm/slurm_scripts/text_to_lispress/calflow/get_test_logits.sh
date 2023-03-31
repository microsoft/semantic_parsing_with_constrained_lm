#!/bin/bash

valid_file="/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2/test_all.jsonl"

for checkpoint in "/brtx/603-nvme1/estengel/calflow_calibration/benchclamp/text_to_calflow/1.0/t5-base-lm-adapt_calflow_last_user_all_0.0001/checkpoint-10000/" "/brtx/604-nvme1/estengel/calflow_calibration/benchclamp/text_to_calflow/1.0/bart-base_calflow_last_user_all_0.0001/checkpoint-10000/" "/brtx/603-nvme1/estengel/calflow_calibration/benchclamp/text_to_calflow/1.0/bart-large_calflow_last_user_all_0.0001/checkpoint-10000"
do
    export CHECKPOINT_DIR=${checkpoint}
    export VALIDATION_FILE=${valid_file}
    sbatch slurm_scripts/text_to_lispress/calflow/get_logits.sh --export 
done
