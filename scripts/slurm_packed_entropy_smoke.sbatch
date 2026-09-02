#!/usr/bin/env bash
#SBATCH --job-name=car-packed-smoke
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
MODEL_PATH="$PROJECT_ROOT/models/derived/Qwen2.5-7B-Instruct-F01-merged-20260901"
cd "$PROJECT_ROOT"
mkdir -p logs/slurm reports

module purge
module load Miniforge3/24.11.3-1
module load CUDA/13.0.0
eval "$(conda shell.bash hook)"
conda activate "$GPU_ENV"
source scripts/cluster_runtime_env.sh
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

{
  echo "job_id=$SLURM_JOB_ID"
  echo "node=$SLURM_JOB_NODELIST"
  scontrol show job -dd "$SLURM_JOB_ID"
  nvidia-smi -L
} >"reports/packed_entropy_smoke_allocation_${SLURM_JOB_ID}.log" 2>&1

python -B scripts/check_packed_entropy.py \
  --model "$MODEL_PATH" \
  --chunk-size 2048 \
  --output "reports/packed_entropy_smoke_${SLURM_JOB_ID}.json"
