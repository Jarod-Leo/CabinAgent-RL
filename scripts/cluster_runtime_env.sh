#!/usr/bin/env bash

: "${PROJECT_ROOT:?PROJECT_ROOT must be set before sourcing cluster_runtime_env.sh}"

cache_root="$PROJECT_ROOT/.cache"
export TMPDIR="$PROJECT_ROOT/cache/tmp"
export HF_HOME="$cache_root/huggingface"
export HF_DATASETS_CACHE="$cache_root/huggingface/datasets"
export XDG_CACHE_HOME="$cache_root"
export VLLM_CACHE_ROOT="$cache_root/vllm"
export TORCH_HOME="$cache_root/torch"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
export TRITON_CACHE_DIR="$cache_root/triton"
export CUDA_CACHE_PATH="$cache_root/cuda"
export FLASHINFER_WORKSPACE_BASE="$PROJECT_ROOT"
# Ray embeds the session directory in AF_UNIX socket names, whose Linux limit is
# 107 bytes. The canonical SSD project path is too long; keep only ephemeral
# Ray sockets/session metadata in a short, Job-ID-scoped /tmp path.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/cabin-ray-${SLURM_JOB_ID:-manual}}"

mkdir -p \
  "$TMPDIR" \
  "$HF_DATASETS_CACHE" \
  "$VLLM_CACHE_ROOT" \
  "$TORCH_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" \
  "$PROJECT_ROOT/.cache/flashinfer" \
  "$RAY_TMPDIR"
