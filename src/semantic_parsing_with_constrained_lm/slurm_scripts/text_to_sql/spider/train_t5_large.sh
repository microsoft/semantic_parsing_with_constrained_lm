#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/spider_train_t5_large.out
#SBATCH --partition=brtx6-10,brtx6
#SBATCH --gpus=6

python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
    --exp-name-pattern 't5-large-lm-adapt_spider_past_none_db_val_all_*'

