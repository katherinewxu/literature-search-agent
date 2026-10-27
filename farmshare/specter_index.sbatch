#!/bin/bash
#SBATCH --job-name=lit-specter
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out

# Encode a frozen store with SPECTER2 (proximity adapter): DATA=synergy-v3
# CPU by default; add `-p gpu --gres=gpu:1` at submission for a GPU.
set -euo pipefail
: "${DATA:?}"
r=/scratch/users/$USER/literature-search-agent
cd ~/literature-search-agent
export HF_HOME="$r/model-cache/huggingface"
mkdir -p "$r/index"
"$r/envs/emb/bin/litsearch" index --data "$r/data/$DATA" --output "$r/index/specter2-$DATA"
