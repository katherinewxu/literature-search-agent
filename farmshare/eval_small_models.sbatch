#!/bin/bash
#SBATCH --job-name=lit-evalsmall
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out

# Serve Qwen3-1.7B with the SFT and GRPO LoRA adapters; evaluate base/SFT/GRPO everywhere.
set -euo pipefail
r=/scratch/users/$USER/literature-search-agent
export MODEL=Qwen/Qwen3-1.7B SERVED=qwen3-1.7b WORKERS=24
export VLLM_ARGS="--enable-lora --max-lora-rank 32 --max-loras 2 --lora-modules qwen3-1.7b-sft=$r/checkpoints/qwen3-1.7b-sft-v1 qwen3-1.7b-grpo=$r/checkpoints/qwen3-1.7b-grpo-v1/final"
export MATRIX_CONFIG="configs/llm/trained_v3.json configs/llm/small_transfer.json configs/llm/small_v3open.json"
bash ~/literature-search-agent/farmshare/llm_matrix.sbatch
