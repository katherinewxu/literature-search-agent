#!/bin/bash
#SBATCH --job-name=lit-later
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out
# LLM-free judge check: are judged-eligible web records cited by later reviews? (OpenAlex only)
set -euo pipefail
oa_file="$HOME/.config/litsearch/openalex.key"
if [ -s "$oa_file" ]; then OPENALEX_API_KEY="$(tr -d '[:space:]' < "$oa_file")"; export OPENALEX_API_KEY; fi
cd ~/literature-search-agent
A=/scratch/users/$USER/literature-search-agent/analysis/webfp
PYTHONPATH=src /scratch/users/$USER/literature-search-agent/envs/ml/bin/python scripts/later_review_check.py $A/judgements_gemini-3.8-flash.jsonl $A/later_review_check.md
