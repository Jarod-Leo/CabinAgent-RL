#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <f10|f11|f12|f13|f14> <start|resume> [run-id]" >&2
  exit 2
fi
experiment="$1"
phase="$2"
requested_run_id="${3:-}"
case "$experiment" in
  f10) config="configs/train/fallback_ablations/vanilla.yaml"; best_checkpoint_enabled=0 ;;
  f11) config="configs/train/fallback_ablations/turn_discount.yaml"; best_checkpoint_enabled=1 ;;
  f12) config="configs/train/fallback_ablations/lata.yaml"; best_checkpoint_enabled=1 ;;
  f13) config="configs/train/fallback_ablations/prm_lite.yaml"; best_checkpoint_enabled=1 ;;
  f14) config="configs/train/fallback_ablations/prm_lite_lata.yaml"; best_checkpoint_enabled=1 ;;
  *) echo "Unknown fallback ablation: $experiment" >&2; exit 2 ;;
esac
PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/projects/cabinagentrlarchive/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
PYTHON_BIN="${PYTHON_BIN:-$GPU_ENV/bin/python}"
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-$ARCHIVE_ROOT/models/derived/Qwen2.5-7B-Instruct-F01-merged-20260901}"
SIMULATOR_MODEL_PATH="${SIMULATOR_MODEL_PATH:-$ARCHIVE_ROOT/models/Qwen/Qwen2.5-72B-Instruct-AWQ}"
target_steps=250
cd "$PROJECT_ROOT"
[[ -x "$PYTHON_BIN" ]] || { echo "Project Python is missing: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$POLICY_MODEL_PATH" ]] || { echo "HDD policy parent is missing: $POLICY_MODEL_PATH" >&2; exit 1; }
[[ -d "$SIMULATOR_MODEL_PATH" ]] || { echo "HDD simulator is missing: $SIMULATOR_MODEL_PATH" >&2; exit 1; }
case "$phase" in
  start)
    [[ -z "$requested_run_id" ]] || run_id="$requested_run_id"
    run_id="${run_id:-${experiment}_formal_$(date -u +%Y%m%dT%H%M%SZ)}"
    "$PYTHON_BIN" -B scripts/init_experiment.py --config "$config" --run-id "$run_id"
    ;;
  resume)
    [[ -n "$requested_run_id" ]] || { echo "resume requires an existing run-id" >&2; exit 2; }
    run_id="$requested_run_id"
    "$PYTHON_BIN" -B scripts/checkpoint_policy.py audit --run-id "$run_id"
    ;;
  *) echo "Unknown phase: $phase" >&2; exit 2 ;;
esac
mkdir -p logs/slurm
exports="ALL,PROJECT_ROOT=$PROJECT_ROOT,ARCHIVE_ROOT=$ARCHIVE_ROOT,GPU_ENV=$GPU_ENV,PYTHON_BIN=$PYTHON_BIN,EXPERIMENT_CONFIG=$config,RUN_ID=$run_id,MAX_TRAINING_STEPS=$target_steps,SAVE_FREQ=50,EVAL_FREQ=50,MAX_ACTOR_CKPT_TO_KEEP=1,MAX_CRITIC_CKPT_TO_KEEP=1,MAX_INFRA_RESTARTS=2,CABIN_BEST_CHECKPOINT_ENABLED=$best_checkpoint_enabled,CABIN_BEST_CHECKPOINT_METRIC=val-core/car_bench/reward/mean@1,WANDB_RUN_ID=$run_id,WANDB_RESUME=allow,POLICY_MODEL_PATH=$POLICY_MODEL_PATH,SIMULATOR_MODEL_PATH=$SIMULATOR_MODEL_PATH,SIMULATOR_GPU_MEMORY_UTILIZATION=${SIMULATOR_GPU_MEMORY_UTILIZATION:-0.86},ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60},ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-16},ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384},ROLLOUT_AGENT_WORKERS=${ROLLOUT_AGENT_WORKERS:-16},PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1},ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-true},ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-true},REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-true},USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-true},ENTROPY_FROM_LOGITS_WITH_CHUNKING=${ENTROPY_FROM_LOGITS_WITH_CHUNKING:-true},ENTROPY_FROM_LOGITS_CHUNK_SIZE=${ENTROPY_FROM_LOGITS_CHUNK_SIZE:-2048}"
job_id="$(sbatch --parsable --time=24:00:00 --job-name="car-${experiment}-full" --export="$exports" scripts/slurm_fallback_grpo.sbatch)"
[[ -n "$job_id" ]] || { echo "Empty fallback ablation Job ID" >&2; exit 1; }
"$PYTHON_BIN" -B scripts/update_experiment_manifest.py --run-id "$run_id" --status submitted --slurm-job-id "$job_id"
printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "continuous-$phase" "$job_id" "$target_steps" >>"experiments/$run_id/submissions.tsv"
echo "FALLBACK_ABLATION_SUBMITTED experiment=$experiment phase=$phase job_id=$job_id run_id=$run_id target_steps=$target_steps"
