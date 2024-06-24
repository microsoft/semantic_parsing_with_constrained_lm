#!/bin/bash 

#SBATCH -o /brtx/601-nvme1/estengel/calflow_calibration/benchclamp/logs/eval_t5_large.out
#SBATCH -p brtx6
#SBATCH --gpus=4

export CHECKPOINT_DIR=/brtx/603-nvme1/estengel/calflow_calibration/benchclamp/text_to_calflow/
#python -m semantic_parsing_with_constrained_lm.run_exp \
#    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
#    --exp-name-pattern 't5-large-lm.*10000_dev_eval'

python -m semantic_parsing_with_constrained_lm.run_exp \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
    --exp-name-pattern 't5-large-lm-adapt_calflow_last_user_all_0.0001_10000_test_eval.*'
    #--exp-name-pattern 't5-large-lm-adapt_calflow_last_user_all_0.0001_10000_test_eval_unconstrained-beam_bs_5'
    
