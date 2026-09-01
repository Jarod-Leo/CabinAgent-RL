#!/usr/bin/env bash
#SBATCH --job-name=car-g00
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:pro6000:2
#SBATCH --constraint=highmem
#SBATCH --time=04:00:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
: "${PIPELINE_ID:?PIPELINE_ID must be exported by the pipeline submitter}"
GATE_RUN_LABEL="${GATE_RUN_LABEL:-G00}"
POLICY_LORA_PATH="${POLICY_LORA_PATH:-}"
POST_GATE_ACTION="${POST_GATE_ACTION:-direct_smoke}"
cd "$PROJECT_ROOT"
source scripts/cluster_runtime_env.sh
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#nodes[@]}" -eq 1 ]] || { echo "Expected one physical node" >&2; exit 2; }
allocated_node="${nodes[0]}"
simulator_port=$((22000 + SLURM_JOB_ID % 10000))
policy_port=$((32000 + SLURM_JOB_ID % 10000))
run_dir="$PROJECT_ROOT/experiments/$GATE_RUN_LABEL"
log_dir="$run_dir/logs"
mkdir -p "$log_dir" "$PROJECT_ROOT/logs/slurm" "$PROJECT_ROOT/reports"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "node=$allocated_node"
  echo "job_nodelist=$SLURM_JOB_NODELIST"
  echo "job_gpus=${SLURM_JOB_GPUS:-unknown}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unknown}"
  scontrol show job -dd "$SLURM_JOB_ID"
  nvidia-smi -L
} >"$log_dir/allocation-${SLURM_JOB_ID}.log" 2>&1

cleanup() {
  if [[ -n "${simulator_step_pid:-}" ]]; then
    kill "$simulator_step_pid" 2>/dev/null || true
    wait "$simulator_step_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

srun --nodes=1 --ntasks=1 --gres=gpu:pro6000:1 --exclusive --gpu-bind=single:1 \
  bash -lc "set -euo pipefail; module purge; module load Miniforge3/24.11.3-1; module load CUDA/12.8.0; eval \"\$(conda shell.bash hook)\"; conda activate '$GPU_ENV'; cd '$PROJECT_ROOT'; echo simulator_cuda_visible_devices=\"\${CUDA_VISIBLE_DEVICES:-unknown}\"; SIMULATOR_MODEL_PATH='$PROJECT_ROOT/models/Qwen/Qwen2.5-72B-Instruct-AWQ' SIMULATOR_PORT='$simulator_port' exec bash scripts/serve_simulator_vllm.sh" \
  >"$log_dir/simulator-${SLURM_JOB_ID}.log" 2>&1 &
simulator_step_pid=$!

srun --nodes=1 --ntasks=1 --gres=gpu:pro6000:1 --exclusive --gpu-bind=single:1 \
  bash -lc "
    set -euo pipefail
    module purge
    module load Miniforge3/24.11.3-1
    module load CUDA/12.8.0
    eval \"\$(conda shell.bash hook)\"
    conda activate '$GPU_ENV'
    cd '$PROJECT_ROOT'
    echo policy_step_cuda_visible_devices=\"\${CUDA_VISIBLE_DEVICES:-unknown}\"
    cleanup_policy() { if [[ -n \"\${policy_pid:-}\" ]]; then kill \"\$policy_pid\" 2>/dev/null || true; wait \"\$policy_pid\" 2>/dev/null || true; fi; }
    trap cleanup_policy EXIT INT TERM
    POLICY_MODEL_PATH='$PROJECT_ROOT/models/Qwen/Qwen2.5-7B-Instruct' POLICY_LORA_PATH='$POLICY_LORA_PATH' POLICY_PORT='$policy_port' bash scripts/serve_policy_vllm.sh >'$log_dir/policy-${SLURM_JOB_ID}.log' 2>&1 &
    policy_pid=\$!
    for attempt in \$(seq 1 180); do
      if curl -fsS 'http://127.0.0.1:$policy_port/health' >/dev/null && curl -fsS 'http://127.0.0.1:$simulator_port/health' >/dev/null; then break; fi
      kill -0 \"\$policy_pid\"
      sleep 5
    done
    curl -fsS 'http://127.0.0.1:$policy_port/health' >/dev/null
    curl -fsS 'http://127.0.0.1:$simulator_port/health' >/dev/null
    python -B scripts/run_direct_rl_gate.py --policy-base-url 'http://127.0.0.1:$policy_port/v1' --simulator-base-url 'http://127.0.0.1:$simulator_port/v1' --policy-adapter '$POLICY_LORA_PATH' --output 'experiments/$GATE_RUN_LABEL/trajectories-${SLURM_JOB_ID}.jsonl' 2>&1 | tee '$log_dir/rollout-${SLURM_JOB_ID}.log'
    python -B scripts/check_rollout_gate.py --input 'experiments/$GATE_RUN_LABEL/trajectories-${SLURM_JOB_ID}.jsonl' --output 'reports/direct_rl_gate_${GATE_RUN_LABEL}_${SLURM_JOB_ID}.json' 2>&1 | tee '$log_dir/gate-check-${SLURM_JOB_ID}.log'
  " >"$log_dir/policy-step-${SLURM_JOB_ID}.log" 2>&1

case "$POST_GATE_ACTION" in
  direct_smoke)
    bash scripts/submit_next_training_stage.sh "$PIPELINE_ID" smoke
    ;;
  none)
    echo "POST_GATE_ACTION=none; gate passed without submitting a training family"
    ;;
  *)
    echo "Unknown POST_GATE_ACTION: $POST_GATE_ACTION" >&2
    exit 2
    ;;
esac
