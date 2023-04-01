#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/cosql_decode_bart_large.out
#SBATCH --partition=brtx6-10,brtx6
#SBATCH --gpus=1

python -m semantic_parsing_with_constrained_lm.run_exp \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
    --exp-name-pattern 'bart-large_cosql_past_all_db_val_all_[01].*dev_eval.*'

python -m semantic_parsing_with_constrained_lm.run_exp \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
    --exp-name-pattern 'bart-large_cosql_past_all_db_val_all_[01].*test_eval.*'
