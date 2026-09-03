#!/usr/bin/env bash
#SBATCH --job-name=car-f10-pilot
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:pro6000:2
#SBATCH --constraint=highmem
#SBATCH --time=06:00:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
PYTHON_BIN="${PYTHON_BIN:-$GPU_ENV/bin/python}"
: "${RUN_ID:?RUN_ID must be exported}"
: "${EXPERIMENT_CONFIG:?EXPERIMENT_CONFIG must be exported}"
[[ -x "$PYTHON_BIN" ]] || {
  echo "Project Python is missing or not executable: $PYTHON_BIN" >&2
  exit 1
}
cd "$PROJECT_ROOT"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#nodes[@]}" -eq 1 ]] || { echo "Expected one physical node" >&2; exit 2; }
simulator_port=$((22000 + SLURM_JOB_ID % 10000))
log_dir="$PROJECT_ROOT/experiments/$RUN_ID/logs"
mkdir -p "$log_dir" "$PROJECT_ROOT/logs/slurm"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "phase=${PILOT_PHASE:-unknown}"
  echo "node=${nodes[0]}"
  echo "job_nodelist=$SLURM_JOB_NODELIST"
  echo "job_gpus=${SLURM_JOB_GPUS:-unknown}"
  echo "max_training_steps=${MAX_TRAINING_STEPS:-unset}"
  echo "simulator_gpu_memory_utilization=${SIMULATOR_GPU_MEMORY_UTILIZATION:-unset}"
  echo "rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-unset}"
  echo "rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-unset}"
  scontrol show job -dd "$SLURM_JOB_ID"
  nvidia-smi -L
} >"$log_dir/allocation-${SLURM_JOB_ID}.log" 2>&1

"$PYTHON_BIN" -B scripts/update_experiment_manifest.py \
  --run-id "$RUN_ID" --status running \
  --slurm-job-id "$SLURM_JOB_ID" --node-list "$SLURM_JOB_NODELIST"

export PROJECT_ROOT GPU_ENV RUN_ID EXPERIMENT_CONFIG
export SIMULATOR_PORT="$simulator_port"
set +e
srun --nodes=1 --ntasks=2 --gpus-per-task=1 --gpu-bind=single:1 \
  --output="$log_dir/role-%t-${SLURM_JOB_ID}.log" \
  --error="$log_dir/role-%t-${SLURM_JOB_ID}.err" \
  bash scripts/run_f10_pilot_task.sh
run_status=$?
set -e

if [[ "$run_status" -eq 0 ]]; then
  final_status=completed
else
  final_status=failed
fi
"$PYTHON_BIN" -B scripts/update_experiment_manifest.py \
  --run-id "$RUN_ID" --status "$final_status" \
  --slurm-job-id "$SLURM_JOB_ID" --node-list "$SLURM_JOB_NODELIST"
exit "$run_status"
