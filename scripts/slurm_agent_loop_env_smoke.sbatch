#!/usr/bin/env bash
#SBATCH --job-name=car-loop-env
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
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
source scripts/cluster_runtime_env.sh
export CAR_BENCH_DATASET_ROOT="$PROJECT_ROOT/data/official/car-bench-dataset"
export SIMULATOR_BASE_URL="http://127.0.0.1:29999/v1"
python -B scripts/check_agent_loop_env.py \
  --config configs/agent_loop/carbench.yaml \
  --output "reports/agent_loop_env_smoke_${SLURM_JOB_ID}.json"
