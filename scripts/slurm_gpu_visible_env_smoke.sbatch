#!/usr/bin/env bash
#SBATCH --job-name=car-gpu-env
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --time=00:10:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
cd "$PROJECT_ROOT"

module purge
module load Miniforge3/24.11.3-1
module load CUDA/12.8.0
eval "$(conda shell.bash hook)"
conda activate "$GPU_ENV"
source scripts/cluster_runtime_env.sh

printf 'before cuda=%s rocr=%s hip=%s\n' \
  "${CUDA_VISIBLE_DEVICES:-unset}" \
  "${ROCR_VISIBLE_DEVICES:-unset}" \
  "${HIP_VISIBLE_DEVICES:-unset}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
python -B scripts/check_gpu_visible_env.py \
  --output "reports/gpu_visible_env_smoke_${SLURM_JOB_ID}.json"
