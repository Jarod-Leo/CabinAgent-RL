#!/usr/bin/env bash
#SBATCH --job-name=car-sim-smoke
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --constraint=highmem
#SBATCH --time=00:30:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
PORT=$((21000 + SLURM_JOB_ID % 10000))
module purge
module load Miniforge3/24.11.3-1
module load CUDA/12.8.0
source activate "$GPU_ENV"
cd "$PROJECT_ROOT"
source scripts/cluster_runtime_env.sh
export SIMULATOR_MODEL_PATH="$PROJECT_ROOT/models/Qwen/Qwen2.5-72B-Instruct-AWQ"
export SIMULATOR_PORT="$PORT"

cleanup() {
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
bash scripts/serve_simulator_vllm.sh >"logs/simulator-smoke-${SLURM_JOB_ID}.server.log" 2>&1 &
server_pid=$!
for _attempt in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Simulator server exited during startup" >&2
    exit 1
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null
python -B scripts/smoke_simulator.py \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --output "reports/simulator_smoke_${SLURM_JOB_ID}.json"
