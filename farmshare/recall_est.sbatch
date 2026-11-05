#!/bin/bash
#SBATCH --job-name=lit-recallest
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out
# End-of-episode recall estimation (Haiku screener via OpenRouter; key never echoed).
set -euo pipefail
key_file="$HOME/.config/litsearch/openrouter.key"
[ -s "$key_file" ] || { echo "missing $key_file" >&2; exit 1; }
MODEL_API_KEY="$(tr -d '[:space:]' < "$key_file")"; export MODEL_API_KEY
cd ~/literature-search-agent
.venv/bin/python scripts/recall_est.py "${CONFIG:-configs/recall_est.json}" "${OUT:-analysis/recall-est}"
.venv/bin/python scripts/recall_calibrate.py "${OUT:-analysis/recall-est}"
