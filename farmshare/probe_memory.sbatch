#!/bin/bash
#SBATCH --job-name=lit-memo
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out

# Closed-book memorization probe: CONFIG=configs/llm/memo_*.json OUTPUT=.../x.jsonl
set -euo pipefail
: "${CONFIG:?}" "${OUTPUT:?}"
key_file="$HOME/.config/litsearch/openrouter.key"
MODEL_API_KEY="$(tr -d '[:space:]' < "$key_file")"
export MODEL_API_KEY
cd ~/literature-search-agent
.venv/bin/litsearch probe-memory --config "$CONFIG" --output "$OUTPUT"
