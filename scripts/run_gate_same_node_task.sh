#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
GPU_ENV="${GPU_ENV:?GPU_ENV is required}"
GATE_RUN_LABEL="${GATE_RUN_LABEL:?GATE_RUN_LABEL is required}"
POLICY_LORA_PATH="${POLICY_LORA_PATH:-}"
SIMULATOR_PORT="${SIMULATOR_PORT:?SIMULATOR_PORT is required}"
POLICY_PORT="${POLICY_PORT:?POLICY_PORT is required}"
log_dir="$PROJECT_ROOT/experiments/$GATE_RUN_LABEL/logs"
done_file="$log_dir/gate-${SLURM_JOB_ID}.done"
cd "$PROJECT_ROOT"

module purge
module load Miniforge3/24.11.3-1
module load CUDA/12.8.0
eval "$(conda shell.bash hook)"
conda activate "$GPU_ENV"
source scripts/cluster_runtime_env.sh
echo "procid=$SLURM_PROCID cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unknown}"

if [[ "$SLURM_PROCID" == "0" ]]; then
  cleanup_simulator() {
    local exit_code=$?
    if [[ -n "${simulator_pid:-}" ]]; then
      kill "$simulator_pid" 2>/dev/null || true
      wait "$simulator_pid" 2>/dev/null || true
    fi
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

cleanup_policy() {
  local exit_code=$?
  touch "$done_file"
  if [[ -n "${policy_pid:-}" ]]; then
    kill "$policy_pid" 2>/dev/null || true
    wait "$policy_pid" 2>/dev/null || true
  fi
  return "$exit_code"
}
trap cleanup_policy EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

POLICY_MODEL_PATH="$PROJECT_ROOT/models/Qwen/Qwen2.5-7B-Instruct" \
  POLICY_LORA_PATH="$POLICY_LORA_PATH" \
  POLICY_PORT="$POLICY_PORT" \
  bash scripts/serve_policy_vllm.sh >"$log_dir/policy-${SLURM_JOB_ID}.log" 2>&1 &
policy_pid=$!

for attempt in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$POLICY_PORT/health" >/dev/null && \
     curl -fsS "http://127.0.0.1:$SIMULATOR_PORT/health" >/dev/null; then
    break
  fi
  kill -0 "$policy_pid"
  sleep 5
done
curl -fsS "http://127.0.0.1:$POLICY_PORT/health" >/dev/null
curl -fsS "http://127.0.0.1:$SIMULATOR_PORT/health" >/dev/null

python -B scripts/run_direct_rl_gate.py \
  --policy-base-url "http://127.0.0.1:$POLICY_PORT/v1" \
  --simulator-base-url "http://127.0.0.1:$SIMULATOR_PORT/v1" \
  --policy-adapter "$POLICY_LORA_PATH" \
  --output "experiments/$GATE_RUN_LABEL/trajectories-${SLURM_JOB_ID}.jsonl" \
  2>&1 | tee "$log_dir/rollout-${SLURM_JOB_ID}.log"
python -B scripts/check_rollout_gate.py \
  --input "experiments/$GATE_RUN_LABEL/trajectories-${SLURM_JOB_ID}.jsonl" \
  --output "reports/direct_rl_gate_${GATE_RUN_LABEL}_${SLURM_JOB_ID}.json" \
  2>&1 | tee "$log_dir/gate-check-${SLURM_JOB_ID}.log"
