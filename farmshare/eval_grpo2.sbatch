#!/bin/bash
#SBATCH --job-name=lit-evalgrpo2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out

# Evaluate the final GRPO v2 adapters (no checkpoint selection) on closed, open and transfer.
set -euo pipefail
r=/scratch/users/$USER/literature-search-agent
c=$r/checkpoints
export MODEL=Qwen/Qwen3-1.7B SERVED=qwen3-1.7b WORKERS=24
export VLLM_ARGS="--enable-lora --max-lora-rank 32 --max-loras 3 --lora-modules grpo2-turn-s1=$c/grpo2-turn-s1/final grpo2-turn-s2=$c/grpo2-turn-s2/final grpo2-fbeta-s1=$c/grpo2-fbeta-s1/final"
export MATRIX_CONFIG="configs/llm/grpo2_v3.json configs/llm/grpo2_transfer.json configs/llm/grpo2_v3open.json"
bash ~/literature-search-agent/farmshare/llm_matrix.sbatch
