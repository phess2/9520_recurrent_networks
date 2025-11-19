#!/bin/bash
module load miniforge
conda activate recurrent-networks

python src/train/train_copy.py "$@"