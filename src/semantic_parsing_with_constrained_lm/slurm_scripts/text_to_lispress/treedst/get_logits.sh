#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/tdst_logits.out
#SBATCH -p brtx6
#SBATCH --gpus=1


python text_to_lispress.py \
    --model_name_or_path ${CHECKPOINT_DIR} \
    --validation_file ${VALIDATION_FILE} \
    --output_dir ${CHECKPOINT_DIR}/outputs \
    --per_device_eval_batch_size 8 \
    --predict_with_generate \
    --get_logits 


