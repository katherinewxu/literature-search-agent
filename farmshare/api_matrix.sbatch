#!/bin/bash
#SBATCH --job-name=lit-api
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out

# Run an API-backed (OpenRouter) LLM matrix on CPU: MATRIX_CONFIG=configs/llm/... [WORKERS=12]
# The key lives in ~/.config/litsearch/openrouter.key (mode 600); it is never echoed or logged.
set -euo pipefail
: "${MATRIX_CONFIG:?}"
key_file="$HOME/.config/litsearch/openrouter.key"
[ -s "$key_file" ] || { echo "missing $key_file" >&2; exit 1; }
MODEL_API_KEY="$(tr -d '[:space:]' < "$key_file")"
export MODEL_API_KEY
cd ~/literature-search-agent
.venv/bin/litsearch matrix --config "$MATRIX_CONFIG" --workers "${WORKERS:-12}" --skip-existing ${SHARD:+--shard $SHARD}
