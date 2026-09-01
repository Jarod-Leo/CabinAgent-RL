#!/usr/bin/env bash
#SBATCH --job-name=car-models
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --time=08:00:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
mkdir -p "$PROJECT_ROOT/cache/tmp" "$PROJECT_ROOT/.cache/huggingface" "$PROJECT_ROOT/models"
export TMPDIR="$PROJECT_ROOT/cache/tmp"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_TELEMETRY=1

module purge
module load Miniforge3/24.11.3-1
source activate "$GPU_ENV"
cd "$PROJECT_ROOT"
python -B scripts/download_models.py \
  --output-root "$PROJECT_ROOT/models" \
  --max-attempts 24 \
  --base-delay 30
