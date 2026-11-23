#!/bin/bash
#SBATCH --job-name=lit-nostop-qwen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out
# Stopping ablation + no-loop baseline for the trained Qwen3-1.7B (GRPO) on the closed graph (no API spend).
set -euo pipefail
r=/scratch/users/$USER/literature-search-agent
export MODEL=Qwen/Qwen3-1.7B SERVED=qwen3-1.7b WORKERS=8
export VLLM_ARGS="--enable-lora --max-lora-rank 32 --max-loras 2 --lora-modules qwen3-1.7b-sft=$r/checkpoints/qwen3-1.7b-sft-v1 qwen3-1.7b-grpo=$r/checkpoints/qwen3-1.7b-grpo-v1/final"
export MATRIX_CONFIG="configs/llm/v3_nostop_qwen.json configs/llm/transfer_nostop_qwen.json"
bash ~/literature-search-agent/farmshare/llm_matrix.sbatch
