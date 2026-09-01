#!/usr/bin/env bash
#SBATCH --job-name=car-f01-merge
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --constraint=highmem
#SBATCH --time=02:00:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$PROJECT_ROOT/models/Qwen/Qwen2.5-7B-Instruct}"
F01_ADAPTER_PATH="${F01_ADAPTER_PATH:-$PROJECT_ROOT/experiments/sft_fallback_full_20260901_stage16/checkpoints/final_adapter}"
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
  echo "base_model=$BASE_MODEL_PATH"
  echo "adapter=$F01_ADAPTER_PATH"
  echo "output=$F01_PARENT_PATH"
  scontrol show job -dd "$SLURM_JOB_ID"
  nvidia-smi -L
} >"logs/f01-parent-merge-allocation-${SLURM_JOB_ID}.log" 2>&1

python -B scripts/merge_lora_parent.py \
  --base-model "$BASE_MODEL_PATH" \
  --adapter "$F01_ADAPTER_PATH" \
  --output "$F01_PARENT_PATH" \
  --expected-rank 16 \
  2>&1 | tee "logs/f01-parent-merge-${SLURM_JOB_ID}.log"
