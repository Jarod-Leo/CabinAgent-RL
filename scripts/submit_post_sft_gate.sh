#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <pipeline-id> <completed-sft-run-id> [gate-label]" >&2
  exit 2
fi

pipeline_id="$1"
sft_run_id="$2"
gate_label="${3:-G03}"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
adapter_path="$PROJECT_ROOT/experiments/$sft_run_id/checkpoints/final_adapter"
cd "$PROJECT_ROOT"

[[ -f "$adapter_path/adapter_config.json" ]] || {
  echo "Missing completed SFT adapter config: $adapter_path/adapter_config.json" >&2
  exit 1
}
[[ -f "$adapter_path/adapter_model.safetensors" ]] || {
  echo "Missing completed SFT adapter weights: $adapter_path/adapter_model.safetensors" >&2
  exit 1
}

dependency_args=()
if [[ -n "${DUAL_GPU_SMOKE_JOB_ID:-}" ]]; then
  dependency_args+=(--dependency="afterok:${DUAL_GPU_SMOKE_JOB_ID}")
fi
job_id="$(sbatch --parsable \
  "${dependency_args[@]}" \
  --job-name="car-${gate_label,,}" \
  --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,PIPELINE_ID=$pipeline_id,GATE_RUN_LABEL=$gate_label,POLICY_LORA_PATH=$adapter_path,POST_GATE_ACTION=none" \
  scripts/slurm_direct_rl_gate_same_node.sbatch)"
[[ -n "$job_id" ]] || { echo "Empty Slurm job ID for $gate_label" >&2; exit 1; }
printf '%s\t%s\t%s\n' "${gate_label,,}" "$job_id" "$adapter_path" >>"experiments/pipelines/${pipeline_id}.tsv"
echo "POST_SFT_GATE_SUBMITTED job_id=$job_id adapter=$adapter_path dependency=${DUAL_GPU_SMOKE_JOB_ID:-none}"
