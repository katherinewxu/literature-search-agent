#!/bin/bash
#SBATCH --job-name=lit-data
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out

set -euo pipefail

project_dir="$HOME/literature-search-agent"
scratch_dir="/scratch/users/$USER/literature-search-agent"
mkdir -p "$scratch_dir/logs" "$scratch_dir/data"
cd "$project_dir"
source .venv/bin/activate

litsearch data build \
  --source synergy \
  --input "$scratch_dir/synergy/raw" \
  --output "$scratch_dir/data/synergy"
litsearch data audit --data "$scratch_dir/data/synergy" \
  > "$scratch_dir/data/audit.json"

