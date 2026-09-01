#!/usr/bin/env bash
#SBATCH --job-name=car-tq-fix
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
cd "$PROJECT_ROOT"

module purge
module load Miniforge3/24.11.3-1
eval "$(conda shell.bash hook)"
conda activate "$GPU_ENV"

# Match veRL's explicit Python requirement and stable-container installation
# style. Avoid dependency resolution changing the frozen CUDA stack.
python -m pip install --no-deps "TransferQueue==0.1.7"
python -B scripts/check_transfer_queue.py \
  --output "reports/transfer_queue_smoke_${SLURM_JOB_ID}.json"
