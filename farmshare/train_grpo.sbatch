#!/bin/bash
#SBATCH --job-name=lit-grpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/%u/literature-search-agent/logs/%x-%j.out

# GRPO: vLLM (rollouts, LoRA hot-loaded each step) and the PEFT trainer share one GPU.
#   CONFIG=configs/train/grpo_*.json BASE=Qwen/Qwen3-1.7B
set -euo pipefail
: "${CONFIG:?}" "${BASE:?}"
r=/scratch/users/$USER/literature-search-agent
cd ~/literature-search-agent
export HF_HOME="$r/model-cache/huggingface"
# FlashInfer JIT-compiles its sampler and FarmShare GPU nodes have no nvcc.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
port=$((8100 + SLURM_JOB_ID % 800))
"$r/envs/ml/bin/vllm" serve "$BASE" --port "$port" --max-model-len 16384 \
  --gpu-memory-utilization "${VLLM_MEM:-0.35}" --enable-prefix-caching \
  --enable-lora --max-lora-rank 64 --max-loras 2 > "$r/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  curl -sf "http://localhost:$port/v1/models" >/dev/null && break
  kill -0 $server 2>/dev/null || { echo "vLLM exited"; tail -40 "$r/logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
MODEL_API_BASE="http://localhost:$port/v1" "$r/envs/ml/bin/litsearch" train-grpo --config "$CONFIG"
