#!/bin/bash
#SBATCH --job-name=lit-sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out

# LoRA SFT of the small policy on rejection-sampled teacher turns: CONFIG=configs/train/...
set -euo pipefail
: "${CONFIG:?}"
r=/scratch/users/$USER/literature-search-agent
cd ~/literature-search-agent
export HF_HOME="$r/model-cache/huggingface"
"$r/envs/ml/bin/litsearch" train-sft --config "$CONFIG"
