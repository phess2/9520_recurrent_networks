#!/bin/bash
#SBATCH --job-name=hydra_copy_sweep
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate /home/ostrow/.conda/envs/recurrent-networks
cd /orcd/data/fiete/001/om2/ostrow/SLT/9520_recurrent_networks

python -m src.train.train_copy --config-name copy -m train.seed=0,1,2 model=lstm,elman,unitary,lru model.hidden_dim=32,64,128 optimizer.lr=1e-2,1e-3,5e-4