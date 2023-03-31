#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/cosql_dev_logits.out
#SBATCH -p brtx6
#SBATCH --gpus=1

# CHECKPOINT_DIR='/srv/local1/estengel/calflow_calibration/benchclamp/lispress_to_text_context/1.0/t5-base-lm-adapt_calflow_last_user_all_0.0001/checkpoint-10000'
# VALIDATION_FILE /brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2/dev_medium.jsonl \

#CHECKPOINT_DIR="/brtx/604-nvme2/estengel/calflow_calibration/benchclamp/1.0/t5-small-lm-adapt_cosql_past_none_db_val_all_0.0001/checkpoint-10000/"
VALIDATION_FILE="/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CoSQL/dev_all.jsonl" 

mkdir -p ${CHECKPOINT_DIR}/outputs

python text_to_lispress.py \
    --model_name_or_path ${BEST_CHECKPOINT} \
    --validation_file ${VALIDATION_FILE} \
    --output_dir ${BEST_CHECKPOINT}/outputs \
    --per_device_eval_batch_size 8 \
    --predict_with_generate \
    --get_logits 


