#!/bin/bash
#SBATCH --job-name=lit-judgeshift
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out
set -euo pipefail
cd ~/literature-search-agent
PYTHONPATH=src /scratch/users/$USER/literature-search-agent/envs/ml/bin/python scripts/judge_shift.py /scratch/users/$USER/literature-search-agent/analysis/webfp/judge_shift.md
