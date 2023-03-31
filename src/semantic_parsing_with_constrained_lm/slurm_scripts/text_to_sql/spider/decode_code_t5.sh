#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/spider_decode_codet5_base.out
#SBATCH --partition=brtx6-10,brtx6
#SBATCH --gpus=2

python -m semantic_parsing_with_constrained_lm.run_exp \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
    --exp-name-pattern 'codet5-base_spider_past_none_db_val_all_[01].*dev_eval.*'

python -m semantic_parsing_with_constrained_lm.run_exp \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
    --exp-name-pattern 'codet5-base_spider_past_none_db_val_all_[01].*test_eval.*'
