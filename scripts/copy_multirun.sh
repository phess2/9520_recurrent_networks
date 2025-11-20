#!/bin/bash
#SBATCH --job-name=hydra_copy_sweep
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail


conda activate recurrent_networks

python -m src.train.train --config-name copy -m train.seed=0,1,2 model.hidden_dim=64,128 optimizer.lr=1e-3,5e-4