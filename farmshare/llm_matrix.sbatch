#!/bin/bash
#SBATCH --job-name=lit-llm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out

# Serve MODEL with vLLM on this GPU, then run MATRIX_CONFIG's LLM cells against it.
#   MODEL=Qwen/Qwen3-32B-AWQ SERVED=qwen3-32b MATRIX_CONFIG=configs/... [WORKERS=16]
#   [VLLM_ARGS="--enable-lora --lora-modules grpo=/path/to/adapter"]
set -euo pipefail
: "${MODEL:?}" "${SERVED:?}" "${MATRIX_CONFIG:?}"
r=/scratch/users/$USER/literature-search-agent
cd ~/literature-search-agent
export HF_HOME="$r/model-cache/huggingface"
# FlashInfer JIT-compiles its sampler and FarmShare GPU nodes have no nvcc.
export VLLM_USE_FLASHINFER_SAMPLER=0
port=$((8100 + SLURM_JOB_ID % 800))
"$r/envs/ml/bin/vllm" serve "$MODEL" --served-model-name "$SERVED" --port "$port" \
  --max-model-len "${MAX_LEN:-16384}" --gpu-memory-utilization 0.90 --enable-prefix-caching \
  ${VLLM_ARGS:-} > "$r/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  curl -sf "http://localhost:$port/v1/models" >/dev/null && break
  kill -0 $server 2>/dev/null || { echo "vLLM exited"; tail -40 "$r/logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
curl -sf "http://localhost:$port/v1/models"
echo
# MATRIX_CONFIG may list several configs separated by spaces; they run in order.
for config in $MATRIX_CONFIG; do
  MODEL_API_BASE="http://localhost:$port/v1" "$r/envs/ml/bin/litsearch" matrix \
    --config "$config" --workers "${WORKERS:-16}" --skip-existing ${SHARD:+--shard $SHARD}
done
