#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/train_t5_large.out
#SBATCH --partition=brtx6-10,brtx6
#SBATCH --gpus=4

python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
    --exp-name-pattern 't5-large-lm-adapt_calflow_last_user_all_*'

