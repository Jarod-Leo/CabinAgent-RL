#!/usr/bin/env bash
#SBATCH --job-name=car-grpo
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:pro6000:2
#SBATCH --constraint=highmem
#SBATCH --time=24:00:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
: "${EXPERIMENT_CONFIG:?EXPERIMENT_CONFIG must be exported by submit_ablation.sh}"
: "${RUN_ID:?RUN_ID must be exported by submit_ablation.sh}"
cd "$PROJECT_ROOT"
source scripts/cluster_runtime_env.sh

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${#nodes[@]}" -ne 1 ]]; then
  echo "Expected exactly one allocated physical node, got ${#nodes[@]}" >&2
  exit 2
fi

allocated_node="${nodes[0]}"
simulator_port=$((20000 + SLURM_JOB_ID % 10000))
run_dir="$PROJECT_ROOT/experiments/$RUN_ID"
mkdir -p "$PROJECT_ROOT/logs/slurm" "$run_dir/logs"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "node=$allocated_node"
  echo "job_nodelist=$SLURM_JOB_NODELIST"
  echo "job_gpus=${SLURM_JOB_GPUS:-unknown}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unknown}"
  scontrol show job -dd "$SLURM_JOB_ID"
  nvidia-smi -L
} >"$run_dir/logs/allocation-${SLURM_JOB_ID}.log" 2>&1

cleanup() {
  local exit_code=$?
  if [[ -n "${simulator_step_pid:-}" ]]; then
    kill "$simulator_step_pid" 2>/dev/null || true
    wait "$simulator_step_pid" 2>/dev/null || true
  fi
  if [[ "$exit_code" -ne 0 ]]; then
    python -B "$PROJECT_ROOT/scripts/update_experiment_manifest.py" \
      --run-id "$RUN_ID" \
      --status failed \
      --slurm-job-id "$SLURM_JOB_ID" \
      --node-list "$SLURM_JOB_NODELIST" || true
  fi
  return "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "RUN_ID=$RUN_ID node=$allocated_node topology=single-node-dual-gpu port=$simulator_port"
python -B scripts/update_experiment_manifest.py \
  --run-id "$RUN_ID" \
  --status running \
  --slurm-job-id "$SLURM_JOB_ID" \
  --node-list "$SLURM_JOB_NODELIST"

srun --nodes=1 --ntasks=1 --gres=gpu:pro6000:1 --exclusive --gpu-bind=single:1 \
  bash -lc "set -euo pipefail; module purge; module load Miniforge3/24.11.3-1; module load CUDA/12.8.0; eval \"\$(conda shell.bash hook)\"; conda activate '$GPU_ENV'; cd '$PROJECT_ROOT'; echo simulator_cuda_visible_devices=\"\${CUDA_VISIBLE_DEVICES:-unknown}\"; SIMULATOR_MODEL_PATH='$PROJECT_ROOT/models/Qwen/Qwen2.5-72B-Instruct-AWQ' SIMULATOR_PORT='$simulator_port' exec bash scripts/serve_simulator_vllm.sh" \
  >"$run_dir/logs/simulator-${SLURM_JOB_ID}.log" 2>&1 &
simulator_step_pid=$!

for attempt in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$simulator_port/health" >/dev/null; then break; fi
  kill -0 "$simulator_step_pid"
  sleep 5
done
curl -fsS "http://127.0.0.1:$simulator_port/health" >/dev/null

srun --nodes=1 --ntasks=1 --gres=gpu:pro6000:1 --exclusive --gpu-bind=single:1 \
  bash -lc "set -euo pipefail; module purge; module load Miniforge3/24.11.3-1; module load CUDA/12.8.0; eval \"\$(conda shell.bash hook)\"; conda activate '$GPU_ENV'; cd '$PROJECT_ROOT'; echo trainer_cuda_visible_devices=\"\${CUDA_VISIBLE_DEVICES:-unknown}\"; export CAR_BENCH_DATASET_ROOT='$PROJECT_ROOT/data/official/car-bench-dataset'; export SIMULATOR_BASE_URL='http://127.0.0.1:$simulator_port/v1'; bash scripts/train_verl_carbench.sh '$EXPERIMENT_CONFIG' '$RUN_ID'" \
  > >(tee "$run_dir/logs/trainer-${SLURM_JOB_ID}.log") 2>&1

python -B scripts/update_experiment_manifest.py \
  --run-id "$RUN_ID" \
  --status completed \
  --slurm-job-id "$SLURM_JOB_ID" \
  --node-list "$SLURM_JOB_NODELIST"

if [[ -n "${NEXT_TRAINING_STAGE:-}" ]]; then
  : "${PIPELINE_ID:?PIPELINE_ID is required when NEXT_TRAINING_STAGE is set}"
  bash scripts/submit_next_training_stage.sh "$PIPELINE_ID" "$NEXT_TRAINING_STAGE"
fi
