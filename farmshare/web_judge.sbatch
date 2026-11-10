#!/bin/bash
#SBATCH --job-name=lit-judge
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out
# LLM eligibility judge on live-web inclusions: JUDGE_CONFIG=configs/judge/... [LIMIT=N batches]
# The key lives in ~/.config/litsearch/openrouter.key (mode 600); it is never echoed or logged.
set -euo pipefail
: "${JUDGE_CONFIG:?}"
key_file="$HOME/.config/litsearch/openrouter.key"
[ -s "$key_file" ] || { echo "missing $key_file" >&2; exit 1; }
MODEL_API_KEY="$(tr -d '[:space:]' < "$key_file")"
export MODEL_API_KEY
cd ~/literature-search-agent
PY=/scratch/users/$USER/literature-search-agent/envs/ml/bin/python
PYTHONPATH=src $PY scripts/web_judge.py judge "$JUDGE_CONFIG" ${LIMIT:+--limit-batches $LIMIT}
PYTHONPATH=src $PY scripts/web_judge.py report "$JUDGE_CONFIG" /scratch/users/$USER/literature-search-agent/analysis/webfp/judge_$(basename "$JUDGE_CONFIG" .json).md
