#!/bin/bash
set -euo pipefail

project_dir="$HOME/literature-search-agent"
config_path="${1:?Usage: submit_one_week.sh /absolute/path/to/matrix.json}"
max_parallel="${MAX_PARALLEL:-16}"
cd "$project_dir"
source .venv/bin/activate

cell_count=$(litsearch matrix --config "$config_path" --count)
if [[ "$cell_count" -lt 1 ]]; then
  echo "Matrix has no cells" >&2
  exit 2
fi

sbatch \
  --array="0-$((cell_count - 1))%${max_parallel}" \
  --export="ALL,MATRIX_CONFIG=$config_path" \
  farmshare/eval_array.sbatch

