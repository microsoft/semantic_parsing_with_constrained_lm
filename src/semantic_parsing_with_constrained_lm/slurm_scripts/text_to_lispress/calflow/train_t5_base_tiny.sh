#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/train_t5_base_tiny.out
#SBATCH -p brtx6
#SBATCH --gpus=1

python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
    --exp-name-pattern 't5-base-lm-adapt_calflow_last_user_low_0_0.0001'

