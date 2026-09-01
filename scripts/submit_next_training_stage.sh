#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <pipeline-id> <smoke|vanilla|turn_discount|lata|prm_lite|prm_lite_lata>" >&2
  exit 2
fi

pipeline_id="$1"
stage="$2"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
cd "$PROJECT_ROOT"

case "$stage" in
  smoke)
    experiment="vanilla"
    base_run_id="smoke_e10_${pipeline_id}"
    next_stage="vanilla"
    walltime="02:00:00"
    extra_exports="MAX_TRAINING_STEPS=2,SAVE_FREQ=1,EVAL_FREQ=1,TRAIN_MAX_SAMPLES=2,VAL_MAX_SAMPLES=2"
    job_name="car-train-smoke"
    ;;
  vanilla)
    experiment="vanilla"
    base_run_id="vanilla_${pipeline_id}"
    next_stage="turn_discount"
    walltime="1-06:00:00"
    extra_exports=""
    job_name="car-vanilla"
    ;;
  turn_discount)
    experiment="turn_discount"
    base_run_id="turn_discount_${pipeline_id}"
    next_stage="lata"
    walltime="1-06:00:00"
    extra_exports=""
    job_name="car-turn-discount"
    ;;
  lata)
    experiment="lata"
    base_run_id="lata_${pipeline_id}"
    next_stage="prm_lite"
    walltime="1-06:00:00"
    extra_exports=""
    job_name="car-lata"
    ;;
  prm_lite)
    experiment="prm_lite"
    base_run_id="prm_lite_${pipeline_id}"
    next_stage="prm_lite_lata"
    walltime="1-06:00:00"
    extra_exports=""
    job_name="car-prm-lite"
    ;;
  prm_lite_lata)
    experiment="prm_lite_lata"
    base_run_id="prm_lite_lata_${pipeline_id}"
    next_stage=""
    walltime="1-06:00:00"
    extra_exports=""
    job_name="car-prm-lata"
    ;;
  *)
    echo "Unknown training stage: $stage" >&2
    exit 2
    ;;
esac

config="configs/train/ablations/${experiment}.yaml"
run_id="$base_run_id"
attempt=0
while [[ -e "experiments/$run_id" ]]; do
  attempt=$((attempt + 1))
  run_id="${base_run_id}_r${attempt}"
done
python -B scripts/init_experiment.py --config "$config" --run-id "$run_id"

exports="ALL,PROJECT_ROOT=$PROJECT_ROOT,EXPERIMENT_CONFIG=$config,RUN_ID=$run_id,PIPELINE_ID=$pipeline_id,NEXT_TRAINING_STAGE=$next_stage"
if [[ -n "$extra_exports" ]]; then
  exports="$exports,$extra_exports"
fi

job_id="$(sbatch --parsable \
  --time="$walltime" \
  --job-name="$job_name" \
  --export="$exports" \
  scripts/slurm_dual_pro6000.sbatch)"
[[ -n "$job_id" ]] || { echo "Empty job ID while submitting $stage" >&2; exit 1; }
python -B scripts/update_experiment_manifest.py \
  --run-id "$run_id" \
  --status submitted \
  --slurm-job-id "$job_id"
printf '%s\t%s\t%s\n' "$stage" "$job_id" "$run_id" >>"experiments/pipelines/${pipeline_id}.tsv"
echo "TRAINING_STAGE_SUBMITTED stage=$stage job_id=$job_id run_id=$run_id"
