#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/train_t5_large_reverse.out
#SBATCH -p brtx6
#SBATCH --gpus=4

python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config_reverse \
    --exp-name-pattern 't5-large-lm-adapt_calflow_last_user_all_0.0001'
