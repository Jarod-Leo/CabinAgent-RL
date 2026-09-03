#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <start|resume> [run-id] [target-steps]" >&2
  exit 2
fi

phase="$1"
requested_run_id="${2:-}"
target_steps="${3:-50}"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
PYTHON_BIN="${PYTHON_BIN:-$GPU_ENV/bin/python}"
config="configs/train/fallback_ablations/vanilla.yaml"
parent_model="$PROJECT_ROOT/models/derived/Qwen2.5-7B-Instruct-F01-merged-20260901"
cd "$PROJECT_ROOT"

[[ -x "$PYTHON_BIN" ]] || {
  echo "Project Python is missing or not executable: $PYTHON_BIN" >&2
  exit 1
}
[[ -f "$parent_model/parent_manifest.json" ]] || {
  echo "Verified merged F01 parent is missing: $parent_model" >&2
  exit 1
}
case "$target_steps" in
  50|100|150|200|250) ;;
  *)
    echo "Formal F10 target must be one of 50, 100, 150, 200, 250" >&2
    exit 2
    ;;
esac
mkdir -p logs/slurm

case "$phase" in
  start)
    [[ "$target_steps" == "50" ]] || {
      echo "A fresh formal F10 run must start at target step 50" >&2
      exit 2
    }
    [[ -z "$requested_run_id" ]] || run_id="$requested_run_id"
    run_id="${run_id:-f10_formal_$(date -u +%Y%m%dT%H%M%SZ)}"
    "$PYTHON_BIN" -B scripts/init_experiment.py --config "$config" --run-id "$run_id"
    ;;
  resume)
    [[ -n "$requested_run_id" ]] || {
      echo "resume requires the existing run-id" >&2
      exit 2
    }
    [[ "$target_steps" != "50" ]] || {
      echo "resume target must be later than step 50" >&2
      exit 2
    }
    run_id="$requested_run_id"
    [[ -f "experiments/$run_id/manifest.json" ]] || {
      echo "Run manifest not found: experiments/$run_id/manifest.json" >&2
      exit 1
    }
    find "experiments/$run_id/checkpoints" -mindepth 1 -type d -name 'global_step_*' -print -quit | grep -q . || {
      echo "No full checkpoint exists for resume: experiments/$run_id/checkpoints" >&2
      exit 1
    }
    ;;
  *)
    echo "Unknown phase: $phase" >&2
    exit 2
    ;;
esac

exports="ALL,PROJECT_ROOT=$PROJECT_ROOT,EXPERIMENT_CONFIG=$config,RUN_ID=$run_id,PILOT_PHASE=formal-$phase,MAX_TRAINING_STEPS=$target_steps,SAVE_FREQ=50,EVAL_FREQ=50,MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1},MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-1},SIMULATOR_GPU_MEMORY_UTILIZATION=${SIMULATOR_GPU_MEMORY_UTILIZATION:-0.86},ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60},ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-16},ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384},ROLLOUT_AGENT_WORKERS=${ROLLOUT_AGENT_WORKERS:-16},PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1},ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-true},ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-true},REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-true},USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-true},ENTROPY_FROM_LOGITS_WITH_CHUNKING=${ENTROPY_FROM_LOGITS_WITH_CHUNKING:-true},ENTROPY_FROM_LOGITS_CHUNK_SIZE=${ENTROPY_FROM_LOGITS_CHUNK_SIZE:-2048}"
job_id="$(sbatch --parsable \
  --time=12:00:00 \
  --job-name="car-f10-${target_steps}" \
  --export="$exports" \
  scripts/slurm_f10_pilot.sbatch)"
[[ -n "$job_id" ]] || { echo "Empty job ID while submitting formal F10" >&2; exit 1; }
"$PYTHON_BIN" -B scripts/update_experiment_manifest.py \
  --run-id "$run_id" --status submitted --slurm-job-id "$job_id"
printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "formal-$phase" "$job_id" "$target_steps" \
  >>"experiments/$run_id/submissions.tsv"
echo "F10_FORMAL_SUBMITTED phase=$phase job_id=$job_id run_id=$run_id target_steps=$target_steps"
