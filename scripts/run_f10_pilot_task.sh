#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
GPU_ENV="${GPU_ENV:?GPU_ENV is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:?EXPERIMENT_CONFIG is required}"
SIMULATOR_PORT="${SIMULATOR_PORT:?SIMULATOR_PORT is required}"
log_dir="$PROJECT_ROOT/experiments/$RUN_ID/logs"
done_file="$log_dir/f10-${SLURM_JOB_ID}.done"
cd "$PROJECT_ROOT"

module purge
module load Miniforge3/24.11.3-1
if [[ "$SLURM_PROCID" == "0" ]]; then
  # Preserve the already validated 72B simulator runtime.
  module load CUDA/12.8.0
else
  # The policy environment uses torch cu130 and its compiled FA2 extension.
  module load CUDA/13.0.0
fi
eval "$(conda shell.bash hook)"
conda activate "$GPU_ENV"
source scripts/cluster_runtime_env.sh
echo "procid=$SLURM_PROCID cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unknown} rocr_visible_devices=${ROCR_VISIBLE_DEVICES:-unset} hip_visible_devices=${HIP_VISIBLE_DEVICES:-unset}"
# cluster02 exports AMD compatibility visibility alongside CUDA even on NVIDIA
# nodes. veRL deliberately rejects both namespaces being active at once.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

bash scripts/monitor_gpu.sh 5 >"$log_dir/gpu-role-${SLURM_PROCID}-${SLURM_JOB_ID}.csv" 2>&1 &
monitor_pid=$!

cleanup_common() {
  local exit_code=$?
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  return "$exit_code"
}

if [[ "$SLURM_PROCID" == "0" ]]; then
  cleanup_simulator() {
    local exit_code=$?
    if [[ -n "${simulator_pid:-}" ]]; then
      kill "$simulator_pid" 2>/dev/null || true
      wait "$simulator_pid" 2>/dev/null || true
    fi
    cleanup_common
    return "$exit_code"
  }
  trap cleanup_simulator EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  SIMULATOR_MODEL_PATH="$PROJECT_ROOT/models/Qwen/Qwen2.5-72B-Instruct-AWQ" \
    SIMULATOR_PORT="$SIMULATOR_PORT" \
    bash scripts/serve_simulator_vllm.sh >"$log_dir/simulator-${SLURM_JOB_ID}.log" 2>&1 &
  simulator_pid=$!
  while [[ ! -e "$done_file" ]]; do
    if ! kill -0 "$simulator_pid" 2>/dev/null; then
      wait "$simulator_pid"
      exit $?
    fi
    sleep 2
  done
  exit 0
fi

if [[ "$SLURM_PROCID" != "1" ]]; then
  echo "Unexpected SLURM_PROCID=$SLURM_PROCID" >&2
  exit 2
fi

cleanup_trainer() {
  local exit_code=$?
  touch "$done_file"
  cleanup_common
  return "$exit_code"
}
trap cleanup_trainer EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for attempt in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$SIMULATOR_PORT/health" >/dev/null; then
    break
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:$SIMULATOR_PORT/health" >/dev/null

export SIMULATOR_BASE_URL="http://127.0.0.1:$SIMULATOR_PORT/v1"
export CAR_BENCH_DATASET_ROOT="$PROJECT_ROOT/data/official/car-bench-dataset"
export CABIN_REWARD_AUDIT_DIR="$PROJECT_ROOT/experiments/$RUN_ID/metrics/rewards-${SLURM_JOB_ID}"
mkdir -p "$CABIN_REWARD_AUDIT_DIR"
python -B scripts/launch_verl.py \
  --config "$EXPERIMENT_CONFIG" \
  --run-id "$RUN_ID" \
  2>&1 | tee "$log_dir/trainer-${SLURM_JOB_ID}.log"
