#!/bin/bash

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/eval_codegen_16b.out
#SBATCH -p ba100
#SBATCH --gpus=4

echo $CUDA_VISIBLE_DEVICES

python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.benchclamp_autoreg_config \
--exp-name-pattern 'codegen-16B_calflow_no_context_low_0_2_dev_eval_constrained_bs_5'

#python -m semantic_parsing_with_constrained_lm.run_exp \
#--config-name semantic_parsing_with_constrained_lm.configs.benchclamp_autoreg_config \
#--exp-name-pattern 'codegen-16B_calflow_no_context_all_2_test_eval_constrained_bs_5'
