#!/usr/bin/env bash
#SBATCH --job-name=car-2gpu-bind
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:pro6000:2
#SBATCH --constraint=highmem
#SBATCH --time=00:10:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
cd "$PROJECT_ROOT"
log_dir="$PROJECT_ROOT/logs/topology"
mkdir -p "$log_dir" "$PROJECT_ROOT/logs/slurm"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#nodes[@]}" -eq 1 ]] || { echo "Expected one physical node" >&2; exit 2; }

{
  echo "job_id=$SLURM_JOB_ID"
  echo "node=${nodes[0]}"
  echo "job_nodelist=$SLURM_JOB_NODELIST"
  echo "job_gpus=${SLURM_JOB_GPUS:-unknown}"
  scontrol show job -dd "$SLURM_JOB_ID"
  nvidia-smi -L
} >"$log_dir/allocation-${SLURM_JOB_ID}.log" 2>&1

srun --nodes=1 --ntasks=2 --gpus-per-task=1 --gpu-bind=single:1 \
  --output="$log_dir/gpu-task-%t-${SLURM_JOB_ID}.log" \
  --error="$log_dir/gpu-task-%t-${SLURM_JOB_ID}.err" \
  bash -lc '
    set -euo pipefail
    echo "procid=${SLURM_PROCID}"
    echo "hostname=$(hostname)"
    echo "slurm_step_gpus=${SLURM_STEP_GPUS:-}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1)
    echo "gpu_uuid=${gpu_uuid}"
    nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader
    sleep 10
  '

step0_log="$log_dir/gpu-task-0-${SLURM_JOB_ID}.log"
step1_log="$log_dir/gpu-task-1-${SLURM_JOB_ID}.log"
step0_uuid="$(sed -n 's/^gpu_uuid=//p' "$step0_log" | head -n 1)"
step1_uuid="$(sed -n 's/^gpu_uuid=//p' "$step1_log" | head -n 1)"
[[ -n "$step0_uuid" && -n "$step1_uuid" ]] || { echo "Missing task GPU UUID" >&2; exit 1; }
[[ "$step0_uuid" != "$step1_uuid" ]] || { echo "Both tasks received the same GPU UUID: $step0_uuid" >&2; exit 1; }
echo "SAME_NODE_DUAL_GPU_OK node=${nodes[0]} task0_uuid=$step0_uuid task1_uuid=$step1_uuid"
