#!/bin/bash

python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.benchclamp_gpt3_config \
--exp-name-pattern 'code-davinci-001_calflow_no_context_all_2_dev_eval_constrained_bs_5'
