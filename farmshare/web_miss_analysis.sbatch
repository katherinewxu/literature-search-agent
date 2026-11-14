#!/bin/bash
#SBATCH --job-name=lit-webmiss
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out
set -euo pipefail
cd ~/literature-search-agent
PYTHONPATH=src /scratch/users/$USER/literature-search-agent/envs/ml/bin/python scripts/web_miss_analysis.py /scratch/users/$USER/literature-search-agent/analysis/webfp
