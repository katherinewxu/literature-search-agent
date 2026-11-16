#!/bin/bash
#SBATCH --job-name=lit-webrefresh
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out
# Recompute every live-web analysis after new runs land (no API calls).
set -euo pipefail
cd ~/literature-search-agent
PY=/scratch/users/$USER/literature-search-agent/envs/ml/bin/python
A=/scratch/users/$USER/literature-search-agent/analysis/webfp
export PYTHONPATH=src
$PY scripts/judge_shift.py $A/judge_shift.md
$PY scripts/web_robust.py $A/web_robust.md
$PY scripts/env_v2_compare.py $A/env_v2_compare.md
$PY scripts/export_web_results.py /scratch/users/$USER/literature-search-agent/results/web
