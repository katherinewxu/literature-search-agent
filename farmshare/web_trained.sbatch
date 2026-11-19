#!/bin/bash
#SBATCH --job-name=lit-webtrained
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out
# Trained Qwen3-1.7B (SFT, GRPO v1 seed 1) on live OpenAlex. Few workers: OpenAlex rate limits.
set -euo pipefail
r=/scratch/users/$USER/literature-search-agent
# Open-web runs query OpenAlex live; the key raises the daily request budget (never echoed).
oa_file="$HOME/.config/litsearch/openalex.key"
if [ -s "$oa_file" ]; then OPENALEX_API_KEY="$(tr -d '[:space:]' < "$oa_file")"; export OPENALEX_API_KEY; fi
export MODEL=Qwen/Qwen3-1.7B SERVED=qwen3-1.7b WORKERS=4
export VLLM_ARGS="--enable-lora --max-lora-rank 32 --max-loras 2 --lora-modules qwen3-1.7b-sft=$r/checkpoints/qwen3-1.7b-sft-v1 qwen3-1.7b-grpo=$r/checkpoints/qwen3-1.7b-grpo-v1/final"
export MATRIX_CONFIG="configs/web/v3_web_trained.json configs/web/transfer_web_trained.json"
bash ~/literature-search-agent/farmshare/llm_matrix.sbatch
