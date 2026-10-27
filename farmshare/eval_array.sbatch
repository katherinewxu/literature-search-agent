#!/bin/bash
#SBATCH --job-name=lit-array
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%A_%a.out

set -euo pipefail

project_dir="$HOME/literature-search-agent"
config_path="${MATRIX_CONFIG:?Set MATRIX_CONFIG to the absolute matrix JSON path}"
cd "$project_dir"
source .venv/bin/activate

export HF_HOME="/scratch/users/$USER/literature-search-agent/model-cache/huggingface"
"${LITSEARCH_BIN:-litsearch}" matrix --config "$config_path" --index "$SLURM_ARRAY_TASK_ID"

