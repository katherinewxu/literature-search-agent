#!/bin/bash
#SBATCH --job-name=lit-build-ext
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out

# Extra SYNERGY+ reviews, disjoint from the frozen 8-review v3 benchmark:
#   NAME=ext-train-v2    KEYS_FILE=configs/reviews/ext_train.txt     SPLIT=train
#   NAME=ext-transfer-v2 KEYS_FILE=configs/reviews/ext_transfer.txt  SPLIT=transfer
# (A file, not a comma list: sbatch --export splits on commas.)
set -euo pipefail
: "${NAME:?}" "${KEYS_FILE:?}" "${SPLIT:?}"
KEYS=$(grep -v '^\s*$' "$HOME/literature-search-agent/$KEYS_FILE" | paste -sd, -)
r=/scratch/users/$USER/literature-search-agent
cd ~/literature-search-agent
for dir in "$r/data/synergy-$NAME-base" "$r/data/synergy-$NAME"; do
  if [ -e "$dir" ]; then echo "refusing to overwrite $dir" >&2; exit 1; fi
done
.venv/bin/litsearch data build --source synergy-source \
  --input ~/.synergy_dataset_source/synergy-dataset-plus \
  --reviews "$KEYS" --split "$SPLIT" --output "$r/data/synergy-$NAME-base"
.venv/bin/litsearch data backfill-abstracts --data "$r/data/synergy-$NAME-base" \
  --output "$r/data/synergy-$NAME" --cache "$r/cache/openalex"
.venv/bin/litsearch data audit --data "$r/data/synergy-$NAME" > "$r/data/synergy-$NAME-audit.json" || true
python3.12 -c "import json;a=json.load(open('$r/data/synergy-$NAME-audit.json'));print('ok',a['ok'],'issues',len(a.get('issues',[])),'warnings',len(a.get('warnings',[])))"
