#!/bin/bash 

#SBATCH -o /dev/null
#SBATCH -p brtx6
#SBATCH --gpus=1

export CHECKPOINT_DIR=/brtx/603-nvme1/estengel/calflow_calibration/benchclamp/text_to_calflow
#python -m semantic_parsing_with_constrained_lm.run_exp \
#    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
#    --exp-name-pattern 't5-small-lm.*10000_dev_eval'

python -m semantic_parsing_with_constrained_lm.run_exp \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
    --exp-name-pattern 't5-small-lm-adapt_calflow_last_user_all_.*dev_eval_.*beam_bs_5'
