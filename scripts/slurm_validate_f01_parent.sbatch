#!/usr/bin/env bash
#SBATCH --job-name=car-f01-validate
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --constraint=highmem
#SBATCH --time=01:00:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
F01_PARENT_PATH="${F01_PARENT_PATH:-$PROJECT_ROOT/models/derived/Qwen2.5-7B-Instruct-F01-merged-20260901}"
cd "$PROJECT_ROOT"

module purge
module load Miniforge3/24.11.3-1
module load CUDA/12.8.0
eval "$(conda shell.bash hook)"
conda activate "$GPU_ENV"
source scripts/cluster_runtime_env.sh

{
  echo "job_id=$SLURM_JOB_ID"
  echo "node=$SLURMD_NODENAME"
  echo "parent=$F01_PARENT_PATH"
  scontrol show job -dd "$SLURM_JOB_ID"
  nvidia-smi -L
} >"logs/f01-parent-validation-allocation-${SLURM_JOB_ID}.log" 2>&1

python -B scripts/validate_merged_parent.py \
  --model "$F01_PARENT_PATH" \
  --output "reports/f01_parent_validation_${SLURM_JOB_ID}.json" \
  2>&1 | tee "logs/f01-parent-validation-${SLURM_JOB_ID}.log"
