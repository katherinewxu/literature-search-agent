#!/bin/bash
#SBATCH --job-name=lit-figs
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out
set -euo pipefail
cd ~/literature-search-agent
/scratch/users/$USER/literature-search-agent/envs/ml/bin/python scripts/report_figures.py analysis/figures
