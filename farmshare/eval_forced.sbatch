#!/bin/bash
#SBATCH --job-name=lit-evalforced
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out

# Trained GRPO (seed 1) with forced citation expansion every 3rd charged action.
set -euo pipefail
r=/scratch/users/$USER/literature-search-agent
export MODEL=Qwen/Qwen3-1.7B SERVED=qwen3-1.7b WORKERS=24
export VLLM_ARGS="--enable-lora --max-lora-rank 32 --max-loras 2 --lora-modules qwen3-1.7b-sft=$r/checkpoints/qwen3-1.7b-sft-v1 qwen3-1.7b-grpo=$r/checkpoints/qwen3-1.7b-grpo-v1/final"
export MATRIX_CONFIG="configs/llm/trained_v3_forced.json"
bash ~/literature-search-agent/farmshare/llm_matrix.sbatch
