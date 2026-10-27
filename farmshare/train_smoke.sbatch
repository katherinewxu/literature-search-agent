#!/bin/bash
#SBATCH --job-name=lit-trainsmoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out

# One-GPU smoke test of the training pipeline (gate requirement): tiny SFT, then 2 GRPO steps.
set -euo pipefail
r=/scratch/users/$USER/literature-search-agent
cd ~/literature-search-agent
nvidia-smi --query-gpu=name,memory.total --format=csv
CONFIG=configs/train/smoke_sft.json bash farmshare/train_sft.sbatch
CONFIG=configs/train/smoke_grpo.json BASE=Qwen/Qwen3-1.7B SLURM_JOB_ID=$SLURM_JOB_ID bash farmshare/train_grpo.sbatch
echo SMOKE_OK
