#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
cd "$PROJECT_ROOT"
if [[ -n "$(squeue --me -h)" && "${ALLOW_ACTIVE_JOBS:-0}" != "1" ]]; then
  echo "Refusing to duplicate the pipeline while this user has active or pending jobs." >&2
  squeue --me
  exit 2
fi

pipeline_id="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p experiments/pipelines
record="experiments/pipelines/${pipeline_id}.tsv"

submit() {
  local stage="$1"
  shift
  local job_id
  if ! job_id="$(sbatch --parsable --kill-on-invalid-dep=yes "$@")"; then
    echo "Failed to submit pipeline stage: $stage" >&2
    return 1
  fi
  [[ -n "$job_id" ]] || { echo "Empty job ID for stage: $stage" >&2; return 1; }
  printf '%s\t%s\n' "$stage" "$job_id" >>"$record"
  echo "$job_id"
}

env_job="$(submit env scripts/setup_gpu_env.sbatch)"
models_job="$(submit models --dependency="afterok:$env_job" scripts/download_models.sbatch)"
data_job="$(submit data --dependency="afterok:$models_job" scripts/prepare_car_data.sbatch)"
sim_job="$(submit simulator_smoke --dependency="afterok:$data_job" scripts/slurm_simulator_smoke.sbatch)"
gate_job="$(submit direct_rl_gate \
  --dependency="afterok:$sim_job" \
  --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,PIPELINE_ID=$pipeline_id" \
  scripts/slurm_direct_rl_gate.sbatch)"

echo "PIPELINE_SUBMITTED id=$pipeline_id jobs=$record"
cat "$record"
