#!/bin/bash
#SBATCH --job-name=lit-opengraph
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out

# Open-graph variant of a frozen store: SRC=synergy-v3 DST=synergy-v3-open
set -euo pipefail
: "${SRC:?}" "${DST:?}"
r=/scratch/users/$USER/literature-search-agent
cd ~/literature-search-agent
# Optional OpenAlex key (own $1/day budget instead of the shared per-IP one); never echoed.
if [ -s "$HOME/.config/litsearch/openalex.key" ]; then
  OPENALEX_API_KEY="$(tr -d '[:space:]' < "$HOME/.config/litsearch/openalex.key")"
  export OPENALEX_API_KEY
fi
.venv/bin/litsearch data open-graph --data "$r/data/$SRC" --output "$r/data/$DST" \
  --cache "$r/cache/openalex" --cap-per-review "${CAP:-5000}" --pool-multiple "${MULT:-1.0}" \
  | python3.12 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps({k:v for k,v in d['open_graph'].items()}, indent=1))"
.venv/bin/litsearch data audit --data "$r/data/$DST" > "$r/data/$DST-audit.json" || true
python3.12 -c "import json;a=json.load(open('$r/data/$DST-audit.json'));print('ok',a['ok'],'issues',len(a.get('issues',[])),'warnings',len(a.get('warnings',[])))"
