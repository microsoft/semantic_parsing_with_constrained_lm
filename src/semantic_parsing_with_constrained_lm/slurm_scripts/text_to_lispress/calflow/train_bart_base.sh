#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/train_bart_base.out
#SBATCH -p brtx6,brtx6-10
#SBATCH --gpus=2

python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
    --exp-name-pattern 'bart-base_calflow_last_user_all_0.0001'

