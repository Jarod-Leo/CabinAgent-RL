#!/usr/bin/env bash
#SBATCH --job-name=car-g03-v2
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:pro6000:2
#SBATCH --constraint=highmem
#SBATCH --time=04:00:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
: "${PIPELINE_ID:?PIPELINE_ID must be exported}"
GATE_RUN_LABEL="${GATE_RUN_LABEL:-G03}"
POLICY_LORA_PATH="${POLICY_LORA_PATH:-}"
POST_GATE_ACTION="${POST_GATE_ACTION:-none}"
cd "$PROJECT_ROOT"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#nodes[@]}" -eq 1 ]] || { echo "Expected one physical node" >&2; exit 2; }
simulator_port=$((22000 + SLURM_JOB_ID % 10000))
policy_port=$((32000 + SLURM_JOB_ID % 10000))
run_dir="$PROJECT_ROOT/experiments/$GATE_RUN_LABEL"
log_dir="$run_dir/logs"
mkdir -p "$log_dir" "$PROJECT_ROOT/logs/slurm" "$PROJECT_ROOT/reports"
done_file="$log_dir/gate-${SLURM_JOB_ID}.done"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "node=${nodes[0]}"
  echo "job_nodelist=$SLURM_JOB_NODELIST"
  echo "job_gpus=${SLURM_JOB_GPUS:-unknown}"
  scontrol show job -dd "$SLURM_JOB_ID"
  nvidia-smi -L
} >"$log_dir/allocation-${SLURM_JOB_ID}.log" 2>&1

export PROJECT_ROOT GPU_ENV GATE_RUN_LABEL POLICY_LORA_PATH
export SIMULATOR_PORT="$simulator_port" POLICY_PORT="$policy_port"
srun --nodes=1 --ntasks=2 --gpus-per-task=1 --gpu-bind=single:1 \
  --output="$log_dir/role-%t-${SLURM_JOB_ID}.log" \
  --error="$log_dir/role-%t-${SLURM_JOB_ID}.err" \
  bash scripts/run_gate_same_node_task.sh

case "$POST_GATE_ACTION" in
  none)
    echo "POST_GATE_ACTION=none; gate completed without submitting training"
    ;;
  *)
    echo "Unsupported post-gate action for fallback gate: $POST_GATE_ACTION" >&2
    exit 2
    ;;
esac
