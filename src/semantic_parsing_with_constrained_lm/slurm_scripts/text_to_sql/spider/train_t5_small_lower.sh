#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/spider_train_t5_small_lower.out
#SBATCH --partition=brtx6
#SBATCH --gpus=1

python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
    --exp-name-pattern 't5-small-lm-adapt_spider_past_none_db_val_all_lower_0.0001'

