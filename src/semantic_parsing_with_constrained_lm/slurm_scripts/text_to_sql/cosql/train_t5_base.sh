#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/cosql_train_t5_base.out
#SBATCH --partition=brtx6-10,brtx6
#SBATCH --gpus=2

python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
    --exp-name-pattern 't5-base-lm-adapt_cosql_past_all_db_val_all_*'

