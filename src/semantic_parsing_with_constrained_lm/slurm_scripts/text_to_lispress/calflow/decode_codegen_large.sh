#!/bin/bash

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/eval_codegen_6b.out
#SBATCH -p ba100
#SBATCH --gpus=3


echo $CUDA_VISIBLE_DEVICES

python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.benchclamp_autoreg_config \
--exp-name-pattern 'codegen-6B_calflow_no_context_all_2_test_eval_constrained_bs_5'
