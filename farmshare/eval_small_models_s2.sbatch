#!/bin/bash
#SBATCH --job-name=lit-evalsmall-s2
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
export VLLM_ARGS="--enable-lora --max-lora-rank 32 --max-loras 2 --lora-modules qwen3-1.7b-sft-s2=$r/checkpoints/qwen3-1.7b-sft-s2 qwen3-1.7b-grpo-s2=$r/checkpoints/qwen3-1.7b-grpo-s2/final"
export MATRIX_CONFIG="configs/llm/trained_v3_s2.json configs/llm/small_transfer_s2.json configs/llm/small_v3open_s2.json"
bash ~/literature-search-agent/farmshare/llm_matrix.sbatch
