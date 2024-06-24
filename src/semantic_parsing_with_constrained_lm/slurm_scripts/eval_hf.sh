#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/train.out
#SBATCH -p brtx6
#SBATCH --gpus=1

CHECKPOINT_DIR='/srv/local1/estengel/calflow_calibration/benchclamp/lispress_to_text_context/1.0/t5-base-lm-adapt_calflow_last_user_all_0.0001/checkpoint-10000'
python lispress_to_text.py \
    --model_name_or_path ${CHECKPOINT_DIR} \
    --validation_file /brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2/dev_medium.jsonl \
    --output_dir ${CHECKPOINT_DIR}/outputs \
    --predict_with_generate \
    --do_eval 


    # --model_name_or_path /brtx/603-nvme1/estengel/calflow_calibration/benchclamp/calflow_to_text/1.0/t5-base-lm-adapt_calflow_last_agent_all_0.0001/checkpoint-10000/ \
