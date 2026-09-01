#!/usr/bin/env bash
#SBATCH --job-name=car-fa2-fix
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --time=02:00:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
MODEL_PATH="$PROJECT_ROOT/models/derived/Qwen2.5-7B-Instruct-F01-merged-20260901"
cd "$PROJECT_ROOT"

module purge
module load Miniforge3/24.11.3-1
module load CUDA/13.0.0
eval "$(conda shell.bash hook)"
conda activate "$GPU_ENV"
source scripts/cluster_runtime_env.sh
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

# veRL's official stable vLLM image pins 2.8.3 and forces a local build. Match
# the cu130 PyTorch toolchain, then limit compilation to Blackwell sm_120.
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=120
export MAX_JOBS=4
python -m pip install --no-build-isolation "flash_attn==2.8.3"
python -B scripts/check_flash_attention.py \
  --model "$MODEL_PATH" \
  --output "reports/flash_attention_smoke_${SLURM_JOB_ID}.json"
