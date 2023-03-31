#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/codegen_logits_2B.out
#SBATCH -p brtx6
#SBATCH --gpus=2

MODEL="codegen-2B" 
TRAIN_FILE="/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2/train_all.jsonl" 
VALIDATION_FILE="/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2/test_all.jsonl" 
#TRAIN_FILE="/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2/train_low_0.jsonl" 
#VALIDATION_FILE="/brtx/601-nvme1/estengel/resources/data/benchclamp/processed/CalFlowV2/dev_low.jsonl" 
OUTPUT_DIR="/brtx/604-nvme2/estengel/calflow_calibration/benchclamp/${MODEL}_calflow"
BEST_CHECKPOINT="/brtx/601-nvme1/estengel/.cache/${MODEL}"

mkdir -p ${OUTPUT_DIR}/outputs 

python text_to_lispress_autoreg.py \
    --model_name_or_path ${BEST_CHECKPOINT} \
    --train_file ${TRAIN_FILE} \
    --validation_file ${VALIDATION_FILE} \
    --output_dir ${OUTPUT_DIR}/outputs \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --get_logits 


