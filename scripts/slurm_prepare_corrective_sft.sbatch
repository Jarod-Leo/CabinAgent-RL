#!/usr/bin/env bash
#SBATCH --job-name=car-f02-data
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --time=00:30:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
PIPELINE_ID="${PIPELINE_ID:-f02_${SLURM_JOB_ID}}"
module purge
module load Miniforge3/24.11.3-1
eval "$(conda shell.bash hook)"
conda activate "$GPU_ENV"
cd "$PROJECT_ROOT"
attempt_dir="$PROJECT_ROOT/data/sft/attempts/$PIPELINE_ID"
stage_log_dir="$PROJECT_ROOT/logs/stages"
mkdir -p "$attempt_dir" "$stage_log_dir" "$PROJECT_ROOT/logs/slurm"
exec > >(tee -a "$stage_log_dir/f02-data-${SLURM_JOB_ID}.out") \
  2> >(tee -a "$stage_log_dir/f02-data-${SLURM_JOB_ID}.err" >&2)

python -B scripts/build_corrective_sft_data.py \
  --successful-input \
    experiments/G00/trajectories-131880.jsonl \
    experiments/G01/trajectories-131930.jsonl \
    experiments/G02/trajectories-131950.jsonl \
    experiments/G03/trajectories-132967.jsonl \
  --failure-input experiments/G03/trajectories-132967.jsonl \
  --train-output "$attempt_dir/corrective_train.jsonl" \
  --val-output "$attempt_dir/corrective_val.jsonl" \
  --report-output "$attempt_dir/corrective_sft_data_report.md" \
  --val-ratio 0.2 \
  --seed 42
cp -p "$attempt_dir/corrective_train.jsonl" data/sft/corrective_train.jsonl
cp -p "$attempt_dir/corrective_val.jsonl" data/sft/corrective_val.jsonl
cp -p "$attempt_dir/corrective_sft_data_report.md" reports/corrective_sft_data_report.md
python -B scripts/check_sft_fallback_data.py \
  --config configs/train/sft_corrective_lora.yaml \
  --output "$attempt_dir/corrective_sft_tokenization.json"
cp -p "$attempt_dir/corrective_sft_tokenization.json" reports/corrective_sft_tokenization.json

if [[ "${AUTO_SUBMIT_SFT_SMOKE:-0}" == "1" ]]; then
  bash scripts/submit_corrective_sft_stage.sh "$PIPELINE_ID" smoke
fi
