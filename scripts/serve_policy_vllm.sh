#!/usr/bin/env bash
set -euo pipefail

MODEL="${POLICY_MODEL_PATH:-models/Qwen/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${POLICY_SERVED_MODEL_NAME:-cabinagent-policy}"
LORA_PATH="${POLICY_LORA_PATH:-}"
HOST="${POLICY_HOST:-0.0.0.0}"
PORT="${POLICY_PORT:-8001}"

args=(python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len "${POLICY_MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "${POLICY_MAX_NUM_SEQS:-16}" \
  --max-num-batched-tokens "${POLICY_MAX_NUM_BATCHED_TOKENS:-8192}" \
  --gpu-memory-utilization "${POLICY_GPU_MEMORY_UTILIZATION:-0.88}" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code)

if [[ -n "$LORA_PATH" ]]; then
  args+=(
    --served-model-name "${POLICY_BASE_SERVED_MODEL_NAME:-cabinagent-policy-base}"
    --enable-lora
    --max-lora-rank 16
    --max-loras 1
    --lora-modules "$SERVED_MODEL_NAME=$LORA_PATH"
  )
else
  args+=(--served-model-name "$SERVED_MODEL_NAME")
fi

exec "${args[@]}"
