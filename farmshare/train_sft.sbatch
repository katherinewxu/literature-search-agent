#!/bin/bash
#SBATCH --job-name=lit-sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out

set -euo pipefail

project_dir="$HOME/literature-search-agent"
scratch_dir="/scratch/users/$USER/literature-search-agent"
mkdir -p "$scratch_dir/logs" "$scratch_dir/model-cache"
cd "$project_dir"
source .venv/bin/activate
export HF_HOME="$scratch_dir/model-cache/huggingface"

litsearch train-sft --config configs/train_sft.json

