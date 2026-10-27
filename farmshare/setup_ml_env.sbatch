#!/bin/bash
#SBATCH --job-name=lit-mlenv
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out

# Two isolated environments on scratch: `ml` (vLLM serving + LoRA/GRPO training) and
# `emb` (SPECTER2 via the adapters library, which pins its own transformers).
set -euo pipefail
scratch_dir="/scratch/users/$USER/literature-search-agent"
project_dir="$HOME/literature-search-agent"
export PIP_CACHE_DIR="$scratch_dir/pip-cache"
mkdir -p "$scratch_dir/envs" "$PIP_CACHE_DIR"

python3.12 -m venv "$scratch_dir/envs/ml"
"$scratch_dir/envs/ml/bin/pip" install -q --upgrade pip uv
"$scratch_dir/envs/ml/bin/uv" pip install --python "$scratch_dir/envs/ml/bin/python" \
  vllm peft trl datasets accelerate matplotlib
"$scratch_dir/envs/ml/bin/uv" pip install --python "$scratch_dir/envs/ml/bin/python" -e "$project_dir"

python3.12 -m venv "$scratch_dir/envs/emb"
"$scratch_dir/envs/emb/bin/pip" install -q --upgrade pip uv
"$scratch_dir/envs/emb/bin/uv" pip install --python "$scratch_dir/envs/emb/bin/python" \
  torch adapters faiss-cpu numpy
"$scratch_dir/envs/emb/bin/uv" pip install --python "$scratch_dir/envs/emb/bin/python" -e "$project_dir"

for env in ml emb; do
  "$scratch_dir/envs/$env/bin/python" -c "import torch, transformers; print('$env', torch.__version__, transformers.__version__)"
done
"$scratch_dir/envs/ml/bin/python" -c "import vllm, trl, peft; print('vllm', vllm.__version__, 'trl', trl.__version__, 'peft', peft.__version__)"
"$scratch_dir/envs/emb/bin/python" -c "import adapters, faiss; print('adapters', adapters.__version__)"
