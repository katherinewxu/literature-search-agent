#!/bin/bash
#SBATCH --job-name=lit-backfill
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out

# Finish the abstract backfill of an already-built extended store (cache-first): NAME=ext-train-v2
set -euo pipefail
: "${NAME:?}"
r=/scratch/users/$USER/literature-search-agent
cd ~/literature-search-agent
if [ -s "$HOME/.config/litsearch/openalex.key" ]; then
  OPENALEX_API_KEY="$(tr -d '[:space:]' < "$HOME/.config/litsearch/openalex.key")"
  export OPENALEX_API_KEY
fi
.venv/bin/litsearch data backfill-abstracts --data "$r/data/synergy-$NAME-base" \
  --output "$r/data/synergy-$NAME" --cache "$r/cache/openalex"
.venv/bin/litsearch data audit --data "$r/data/synergy-$NAME" > "$r/data/synergy-$NAME-audit.json" || true
python3.12 -c "import json;a=json.load(open('$r/data/synergy-$NAME-audit.json'));print('ok',a['ok'],'warnings',len(a.get('warnings',[])))"
