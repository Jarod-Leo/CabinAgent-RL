#!/usr/bin/env bash
set -euo pipefail

MODEL="${SIMULATOR_MODEL_PATH:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
SERVED_MODEL_NAME="${SIMULATOR_SERVED_MODEL_NAME:-cabinagent-user-simulator}"
HOST="${SIMULATOR_HOST:-0.0.0.0}"
PORT="${SIMULATOR_PORT:-8000}"
MAX_MODEL_LEN="${SIMULATOR_MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${SIMULATOR_MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${SIMULATOR_MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${SIMULATOR_GPU_MEMORY_UTILIZATION:-0.92}"
QUANTIZATION="${SIMULATOR_QUANTIZATION:-awq_marlin}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --quantization "$QUANTIZATION" \
  --dtype auto \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --trust-remote-code
