#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <start|resume> [run-id]" >&2
  exit 2
fi

phase="$1"
requested_run_id="${2:-}"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
config="configs/train/fallback_ablations/vanilla.yaml"
parent_model="$PROJECT_ROOT/models/derived/Qwen2.5-7B-Instruct-F01-merged-20260901"
cd "$PROJECT_ROOT"

[[ -f "$parent_model/parent_manifest.json" ]] || {
  echo "Verified merged F01 parent is missing: $parent_model" >&2
  exit 1
}
mkdir -p logs/slurm

case "$phase" in
  start)
    [[ -z "$requested_run_id" ]] || run_id="$requested_run_id"
    run_id="${run_id:-f10_pilot_$(date -u +%Y%m%dT%H%M%SZ)}"
    python -B scripts/init_experiment.py --config "$config" --run-id "$run_id"
    max_steps=5
    save_freq=5
    job_name="car-f10-pilot"
    ;;
  resume)
    [[ -n "$requested_run_id" ]] || {
      echo "resume requires the existing run-id" >&2
      exit 2
    }
    run_id="$requested_run_id"
    [[ -f "experiments/$run_id/manifest.json" ]] || {
      echo "Run manifest not found: experiments/$run_id/manifest.json" >&2
      exit 1
    }
    find "experiments/$run_id/checkpoints" -mindepth 1 -print -quit | grep -q . || {
      echo "No checkpoint exists for resume: experiments/$run_id/checkpoints" >&2
      exit 1
    }
    max_steps=6
    # Keep the validated step-5 recovery point without transiently creating a
    # second ~30 GB checkpoint for this one-step resume validation.
    save_freq=-1
    job_name="car-f10-resume"
    ;;
  *)
    echo "Unknown phase: $phase" >&2
    exit 2
    ;;
esac

exports="ALL,PROJECT_ROOT=$PROJECT_ROOT,EXPERIMENT_CONFIG=$config,RUN_ID=$run_id,PILOT_PHASE=$phase,MAX_TRAINING_STEPS=$max_steps,SAVE_FREQ=$save_freq,EVAL_FREQ=5,MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1},MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-1},SIMULATOR_GPU_MEMORY_UTILIZATION=${SIMULATOR_GPU_MEMORY_UTILIZATION:-0.86},ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60},ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-16},ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384},ROLLOUT_AGENT_WORKERS=${ROLLOUT_AGENT_WORKERS:-16},PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1},ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-true},ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-true},REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-true},USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-true},ENTROPY_FROM_LOGITS_WITH_CHUNKING=${ENTROPY_FROM_LOGITS_WITH_CHUNKING:-true},ENTROPY_FROM_LOGITS_CHUNK_SIZE=${ENTROPY_FROM_LOGITS_CHUNK_SIZE:-2048}"
job_id="$(sbatch --parsable \
  --job-name="$job_name" \
  --export="$exports" \
  scripts/slurm_f10_pilot.sbatch)"
[[ -n "$job_id" ]] || { echo "Empty job ID while submitting F10 $phase" >&2; exit 1; }
python -B scripts/update_experiment_manifest.py \
  --run-id "$run_id" --status submitted --slurm-job-id "$job_id"
printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$phase" "$job_id" "$max_steps" \
  >>"experiments/$run_id/submissions.tsv"
echo "F10_PILOT_SUBMITTED phase=$phase job_id=$job_id run_id=$run_id max_steps=$max_steps"
