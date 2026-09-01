#!/usr/bin/env bash
#SBATCH --job-name=car-sft-fallback
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --constraint=highmem
#SBATCH --time=08:00:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
: "${RUN_ID:?RUN_ID must be exported by submit_sft_fallback_stage.sh}"
: "${DIRECT_RL_GATE_REPORT:?DIRECT_RL_GATE_REPORT must identify the failed gate}"
SFT_CONFIG="${SFT_CONFIG:-configs/train/sft_fallback_lora.yaml}"
cd "$PROJECT_ROOT"
source scripts/cluster_runtime_env.sh
run_dir="$PROJECT_ROOT/experiments/$RUN_ID"
mkdir -p "$run_dir/logs" "$PROJECT_ROOT/logs/slurm"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "node=$SLURM_JOB_NODELIST"
  echo "job_gpus=${SLURM_JOB_GPUS:-unknown}"
  scontrol show job -dd "$SLURM_JOB_ID"
  nvidia-smi -L
} >"$run_dir/logs/allocation-${SLURM_JOB_ID}.log" 2>&1

cleanup() {
  local exit_code=$?
  if [[ "$exit_code" -ne 0 ]]; then
    envs/cabinagentrl/bin/python -B scripts/update_experiment_manifest.py \
      --run-id "$RUN_ID" --status failed --slurm-job-id "$SLURM_JOB_ID" \
      --node-list "$SLURM_JOB_NODELIST" || true
  fi
  return "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

envs/cabinagentrl/bin/python -B scripts/update_experiment_manifest.py \
  --run-id "$RUN_ID" --status running --slurm-job-id "$SLURM_JOB_ID" \
  --node-list "$SLURM_JOB_NODELIST"

trainer_args=()
if [[ -n "${MAX_TRAINING_STEPS:-}" ]]; then
  trainer_args+=(--max-steps "$MAX_TRAINING_STEPS")
fi
if [[ -n "${TRAIN_MAX_RECORDS:-}" ]]; then
  trainer_args+=(--train-max-records "$TRAIN_MAX_RECORDS")
fi

srun --nodes=1 --ntasks=1 --gres=gpu:pro6000:1 --exclusive --gpu-bind=single:1 \
  bash -lc "set -euo pipefail; module purge; module load Miniforge3/24.11.3-1; module load CUDA/12.8.0; eval \"\$(conda shell.bash hook)\"; conda activate '$GPU_ENV'; cd '$PROJECT_ROOT'; echo trainer_cuda_visible_devices=\"\${CUDA_VISIBLE_DEVICES:-unknown}\"; export DIRECT_RL_GATE_REPORT='$DIRECT_RL_GATE_REPORT'; export SFT_CONFIG='$SFT_CONFIG'; bash scripts/train_sft_fallback.sh '$RUN_ID' ${trainer_args[*]}" \
  > >(tee "$run_dir/logs/trainer-${SLURM_JOB_ID}.log") 2>&1

envs/cabinagentrl/bin/python -B scripts/update_experiment_manifest.py \
  --run-id "$RUN_ID" --status completed --slurm-job-id "$SLURM_JOB_ID" \
  --node-list "$SLURM_JOB_NODELIST"

if [[ -n "${NEXT_SFT_STAGE:-}" ]]; then
  : "${PIPELINE_ID:?PIPELINE_ID is required when NEXT_SFT_STAGE is set}"
  bash "${SFT_STAGE_SUBMITTER:-scripts/submit_sft_fallback_stage.sh}" "$PIPELINE_ID" "$NEXT_SFT_STAGE"
elif [[ "${POST_SFT_ACTION:-none}" == "gate" ]]; then
  : "${PIPELINE_ID:?PIPELINE_ID is required when POST_SFT_ACTION=gate}"
  bash scripts/submit_post_sft_gate.sh "$PIPELINE_ID" "$RUN_ID" "${POST_GATE_LABEL:-G03}"
fi
