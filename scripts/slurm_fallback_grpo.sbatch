#!/usr/bin/env bash
#SBATCH --job-name=car-fallback-grpo
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:pro6000:2
#SBATCH --constraint=highmem
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
PYTHON_BIN="${PYTHON_BIN:-$GPU_ENV/bin/python}"
: "${RUN_ID:?RUN_ID must be exported}"
: "${EXPERIMENT_CONFIG:?EXPERIMENT_CONFIG must be exported}"
: "${MAX_TRAINING_STEPS:?MAX_TRAINING_STEPS must be exported}"
: "${POLICY_MODEL_PATH:?POLICY_MODEL_PATH must be exported}"
: "${SIMULATOR_MODEL_PATH:?SIMULATOR_MODEL_PATH must be exported}"
restart_count="${SLURM_RESTART_COUNT:-0}"
max_restarts="${MAX_INFRA_RESTARTS:-2}"
(( restart_count <= max_restarts )) || {
  echo "Infrastructure restart limit exceeded: $restart_count > $max_restarts" >&2
  exit 1
}
[[ -x "$PYTHON_BIN" ]] || { echo "Project Python is missing: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$POLICY_MODEL_PATH" ]] || { echo "Policy model is missing: $POLICY_MODEL_PATH" >&2; exit 1; }
[[ -d "$SIMULATOR_MODEL_PATH" ]] || { echo "Simulator model is missing: $SIMULATOR_MODEL_PATH" >&2; exit 1; }
cd "$PROJECT_ROOT"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#nodes[@]}" -eq 1 ]] || { echo "Expected one physical node" >&2; exit 2; }
simulator_port=$((24000 + SLURM_JOB_ID % 10000))
log_dir="$PROJECT_ROOT/experiments/$RUN_ID/logs"
mkdir -p "$log_dir" "$PROJECT_ROOT/logs/slurm" "$PROJECT_ROOT/reports/storage"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "node=${nodes[0]}"
  echo "restart_count=$restart_count"
  echo "max_training_steps=$MAX_TRAINING_STEPS"
  echo "policy_model_path=$POLICY_MODEL_PATH"
  echo "simulator_model_path=$SIMULATOR_MODEL_PATH"
  scontrol show job -dd "$SLURM_JOB_ID"
  nvidia-smi -L
} >"$log_dir/allocation-${SLURM_JOB_ID}.log" 2>&1

"$PYTHON_BIN" -B scripts/update_experiment_manifest.py \
  --run-id "$RUN_ID" --status running \
  --slurm-job-id "$SLURM_JOB_ID" --node-list "$SLURM_JOB_NODELIST"
export PROJECT_ROOT GPU_ENV PYTHON_BIN RUN_ID EXPERIMENT_CONFIG POLICY_MODEL_PATH SIMULATOR_MODEL_PATH
export SIMULATOR_PORT="$simulator_port"
export GRPO_DONE_FILE="$log_dir/grpo-${SLURM_JOB_ID}-restart-${restart_count}.done"
set +e
srun --nodes=1 --ntasks=2 --gpus-per-task=1 --gpu-bind=single:1 \
  --output="$log_dir/role-%t-${SLURM_JOB_ID}.log" \
  --error="$log_dir/role-%t-${SLURM_JOB_ID}.err" \
  bash scripts/run_fallback_grpo_task.sh
run_status=$?
set -e

if [[ "$run_status" -eq 0 ]]; then
  checkpoint_args=(audit-series --run-id "$RUN_ID" --expected-step "$MAX_TRAINING_STEPS" --interval "${SAVE_FREQ:-50}")
  report_name="checkpoint-best-${SLURM_JOB_ID}.json"
  if [[ "${CABIN_BEST_CHECKPOINT_ENABLED:-0}" != "1" ]]; then
    checkpoint_args=(prune --run-id "$RUN_ID" --expected-step "$MAX_TRAINING_STEPS" --apply)
    report_name="checkpoint-prune-${SLURM_JOB_ID}.json"
  fi
  if ! "$PYTHON_BIN" -B scripts/checkpoint_policy.py "${checkpoint_args[@]}" \
    --report "$PROJECT_ROOT/reports/storage/$report_name"; then
    echo "Training completed but checkpoint postcondition failed" >&2
    run_status=1
  fi
fi
if [[ "$run_status" -eq 0 ]]; then final_status=completed; else final_status=failed; fi
"$PYTHON_BIN" -B scripts/update_experiment_manifest.py \
  --run-id "$RUN_ID" --status "$final_status" \
  --slurm-job-id "$SLURM_JOB_ID" --node-list "$SLURM_JOB_NODELIST"
exit "$run_status"
