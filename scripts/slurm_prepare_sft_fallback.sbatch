#!/usr/bin/env bash
#SBATCH --job-name=car-sft-data
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --time=00:30:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
PIPELINE_ID="${PIPELINE_ID:-manual_${SLURM_JOB_ID}}"
module purge
module load Miniforge3/24.11.3-1
eval "$(conda shell.bash hook)"
conda activate "$GPU_ENV"
cd "$PROJECT_ROOT"
attempt_dir="$PROJECT_ROOT/data/sft/attempts/$PIPELINE_ID"
stage_log_dir="$PROJECT_ROOT/logs/stages"
mkdir -p "$attempt_dir" "$stage_log_dir" "$PROJECT_ROOT/data/sft/archive"
exec > >(tee -a "$stage_log_dir/sft-data-${SLURM_JOB_ID}.out") \
  2> >(tee -a "$stage_log_dir/sft-data-${SLURM_JOB_ID}.err" >&2)

legacy_dir="$PROJECT_ROOT/data/sft/archive/pre_object_arguments_20260901"
if [[ ! -d "$legacy_dir" && -f data/sft/fallback_train.jsonl ]]; then
  mkdir -p "$legacy_dir"
  cp -p data/sft/fallback_train.jsonl "$legacy_dir/"
  if [[ -f data/sft/fallback_val.jsonl ]]; then
    cp -p data/sft/fallback_val.jsonl "$legacy_dir/"
  fi
fi

python -B scripts/build_sft_data.py \
  --input \
    experiments/G00/trajectories-131880.jsonl \
    experiments/G01/trajectories-131930.jsonl \
    experiments/G02/trajectories-131950.jsonl \
  --train-output "$attempt_dir/fallback_train.jsonl" \
  --val-output "$attempt_dir/fallback_val.jsonl" \
  --report-output "$attempt_dir/sft_fallback_data_report.md" \
  --val-ratio 0.2 \
  --seed 42
cp -p "$attempt_dir/fallback_train.jsonl" data/sft/fallback_train.jsonl
cp -p "$attempt_dir/fallback_val.jsonl" data/sft/fallback_val.jsonl
cp -p "$attempt_dir/sft_fallback_data_report.md" reports/sft_fallback_data_report.md
python -B scripts/check_sft_fallback_data.py \
  --config configs/train/sft_fallback_lora.yaml \
  --output "$attempt_dir/sft_fallback_tokenization.json"
cp -p "$attempt_dir/sft_fallback_tokenization.json" reports/sft_fallback_tokenization.json

if [[ "${AUTO_SUBMIT_SFT_SMOKE:-0}" == "1" ]]; then
  bash scripts/submit_sft_fallback_stage.sh "$PIPELINE_ID" smoke
fi
