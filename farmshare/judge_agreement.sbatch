#!/bin/bash
#SBATCH --job-name=lit-agree
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out
set -euo pipefail
cd ~/literature-search-agent
A=/scratch/users/$USER/literature-search-agent/analysis/webfp
PYTHONPATH=src /scratch/users/$USER/literature-search-agent/envs/ml/bin/python scripts/judge_agreement.py $A/judgements_gemini-3.8-flash.jsonl $A/judgements_gpt-5.6-luna.jsonl $A/judge_agreement.md
