#!/usr/bin/env bash
#SBATCH --job-name=car-f10-adapter
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
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/projects/cabinagentrlarchive/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
PYTHON_BIN="${PYTHON_BIN:-$GPU_ENV/bin/python}"
SOURCE_CHECKPOINT="$ARCHIVE_ROOT/experiments/f10_formal_20260903_stage19/checkpoints/global_step_50"
PARENT_MODEL="$ARCHIVE_ROOT/models/derived/Qwen2.5-7B-Instruct-F01-merged-20260901"
OUTPUT_ADAPTER="$ARCHIVE_ROOT/adapters/f10_vanilla_best_step_50"
REPORT="$PROJECT_ROOT/reports/storage/f10-best-adapter-${SLURM_JOB_ID}.json"

cd "$PROJECT_ROOT"
module purge
module load Miniforge3/24.11.3-1 CUDA/13.0.0
eval "$(conda shell.bash hook)"
conda activate "$GPU_ENV"
source scripts/cluster_runtime_env.sh
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

{
  echo "job_id=$SLURM_JOB_ID"
  echo "node=$SLURM_JOB_NODELIST"
  echo "source_checkpoint=$SOURCE_CHECKPOINT"
  echo "parent_model=$PARENT_MODEL"
  echo "output_adapter=$OUTPUT_ADAPTER"
  nvidia-smi -L
} >"$PROJECT_ROOT/logs/f10-best-adapter-allocation-${SLURM_JOB_ID}.log" 2>&1

"$PYTHON_BIN" -B scripts/export_verl_lora.py \
  --checkpoint "$SOURCE_CHECKPOINT" \
  --parent "$PARENT_MODEL" \
  --output "$OUTPUT_ADAPTER" \
  --step 50 \
  --metric-value 0.269231 \
  --source-slurm-job 136868 \
  --report "$REPORT"
