#!/usr/bin/env bash
#SBATCH --job-name=car-env
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --time=02:00:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
mkdir -p "$PROJECT_ROOT/cache/tmp" "$PROJECT_ROOT/cache/pip" "$PROJECT_ROOT/.cache/huggingface"
export TMPDIR="$PROJECT_ROOT/cache/tmp"
export PIP_CACHE_DIR="$PROJECT_ROOT/cache/pip"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"

module purge
module load Miniforge3/24.11.3-1
module load CUDA/12.8.0
source activate

if [[ ! -x "$GPU_ENV/bin/python" ]]; then
  conda create -y -p "$GPU_ENV" python=3.11 pip
fi
source activate "$GPU_ENV"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$PROJECT_ROOT/requirements-gpu.txt"
python -m pip install -e "$PROJECT_ROOT/data/official/car-bench"
python -m pip check
python - <<'PY'
import importlib.metadata as metadata
import torch

for name in ("torch", "vllm", "verl", "transformers", "datasets", "car_bench"):
    print(f"{name}={metadata.version(name)}")
print(f"cuda_build={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
PY
echo "GPU_ENV_READY=$GPU_ENV"

