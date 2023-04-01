#!/bin/bash 

#SBATCH -o /dev/null
#SBATCH -p brtx6
#SBATCH --gpus=1

# CHECKPOINT_DIR='/srv/local1/estengel/calflow_calibration/benchclamp/lispress_to_text_context/1.0/t5-base-lm-adapt_calflow_last_user_all_0.0001/checkpoint-10000'
# VALIDATION_FILE /brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2/dev_medium.jsonl \

python text_to_lispress.py \
    --model_name_or_path ${CHECKPOINT_DIR} \
    --validation_file ${VALIDATION_FILE} \
    --output_dir ${CHECKPOINT_DIR}/outputs \
    --predict_with_generate \
    --do_eval 


