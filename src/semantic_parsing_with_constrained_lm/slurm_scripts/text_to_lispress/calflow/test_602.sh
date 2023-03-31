#!/bin/bash

#SBATCH -o /home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/logs/602-test.out
#SBATCH -p ba100
#SBATCH --gpus=1

echo $CUDA_VISIBLE_DEVICES
python scripts/602_test.py
