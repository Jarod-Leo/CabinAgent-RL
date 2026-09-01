#!/usr/bin/env bash
#SBATCH --job-name=car-data
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --time=00:30:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
module purge
module load Miniforge3/24.11.3-1
source activate "$GPU_ENV"
cd "$PROJECT_ROOT"
python -B scripts/build_carbench_parquet.py \
  --dataset-root data/official/car-bench-dataset \
  --output-dir data/processed/carbench

