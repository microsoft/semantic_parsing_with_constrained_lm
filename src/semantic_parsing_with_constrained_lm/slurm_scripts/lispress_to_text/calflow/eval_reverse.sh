#!/bin/bash 

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/train.out
#SBATCH -p brtx6
#SBATCH --gpus=1


python -m semantic_parsing_with_constrained_lm.run_exp \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config_reverse \
    --exp-name-pattern 't5-base-lm-adapt_calflow_last_agent_all_.*'
