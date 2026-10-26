#!/bin/bash
#SBATCH --job-name=lit-eval
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out

set -euo pipefail

project_dir="$HOME/literature-search-agent"
scratch_dir="/scratch/users/$USER/literature-search-agent"
mkdir -p "$scratch_dir/logs" "$scratch_dir/runs/mvp"
cd "$project_dir"
source .venv/bin/activate

litsearch matrix --config configs/mvp_matrix.json
litsearch evaluate \
  --run-dir "$scratch_dir/runs/mvp" \
  --output "$scratch_dir/runs/mvp/summary.json"

