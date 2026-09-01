#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <pipeline-id> <smoke|full>" >&2
  exit 2
fi

pipeline_id="$1"
stage="$2"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
gate_report="${DIRECT_RL_GATE_REPORT:-reports/direct_rl_gate_G02_131950.json}"
PYTHON="$PROJECT_ROOT/envs/cabinagentrl/bin/python"
cd "$PROJECT_ROOT"

case "$stage" in
  smoke)
    base_run_id="sft_fallback_smoke_${pipeline_id}"
    job_name="car-sft-smoke"
    walltime="02:00:00"
    next_stage="full"
    post_sft_action="none"
    extra_exports="MAX_TRAINING_STEPS=2,TRAIN_MAX_RECORDS=4"
    ;;
  full)
    base_run_id="sft_fallback_full_${pipeline_id}"
    job_name="car-sft-fallback"
    walltime="08:00:00"
    next_stage=""
    post_sft_action="gate"
    extra_exports="MAX_TRAINING_STEPS=-1,TRAIN_MAX_RECORDS=-1"
    ;;
  *)
    echo "Unknown fallback stage: $stage" >&2
    exit 2
    ;;
esac

run_id="$base_run_id"
attempt=0
while [[ -e "experiments/$run_id" ]]; do
  attempt=$((attempt + 1))
  run_id="${base_run_id}_r${attempt}"
done
"$PYTHON" -B scripts/init_experiment.py \
  --config configs/train/sft_fallback_lora.yaml \
  --run-id "$run_id"

exports="ALL,PROJECT_ROOT=$PROJECT_ROOT,RUN_ID=$run_id,PIPELINE_ID=$pipeline_id,DIRECT_RL_GATE_REPORT=$gate_report,SFT_CONFIG=configs/train/sft_fallback_lora.yaml,NEXT_SFT_STAGE=$next_stage,POST_SFT_ACTION=$post_sft_action,POST_GATE_LABEL=G03"
if [[ -n "$extra_exports" ]]; then
  exports="$exports,$extra_exports"
fi
job_id="$(sbatch --parsable --time="$walltime" --job-name="$job_name" --export="$exports" scripts/slurm_sft_fallback.sbatch)"
[[ -n "$job_id" ]] || { echo "Empty Slurm job ID for fallback $stage" >&2; exit 1; }
"$PYTHON" -B scripts/update_experiment_manifest.py \
  --run-id "$run_id" --status submitted --slurm-job-id "$job_id"
printf 'sft_%s\t%s\t%s\n' "$stage" "$job_id" "$run_id" >>"experiments/pipelines/${pipeline_id}.tsv"
echo "SFT_FALLBACK_SUBMITTED stage=$stage job_id=$job_id run_id=$run_id"
