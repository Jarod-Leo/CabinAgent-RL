#!/usr/bin/env bash
#SBATCH --job-name=cabin-archive
#SBATCH --partition=cluster02
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --output=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.out
#SBATCH --error=/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/projects/cabinagentrlarchive/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
PYTHON_BIN="${PYTHON_BIN:-$GPU_ENV/bin/python}"
: "${ARCHIVE_RELATIVE_PATHS:?ARCHIVE_RELATIVE_PATHS is required}"
: "${ARCHIVE_BATCH_ID:?ARCHIVE_BATCH_ID is required}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing project Python: $PYTHON_BIN" >&2; exit 1; }
cd "$PROJECT_ROOT"
IFS=':' read -r -a relative_paths <<<"$ARCHIVE_RELATIVE_PATHS"
arguments=()
for relative in "${relative_paths[@]}"; do
  arguments+=(--relative "$relative")
done
"$PYTHON_BIN" -B scripts/archive_storage.py \
  --source-root "$PROJECT_ROOT" \
  --archive-root "$ARCHIVE_ROOT" \
  --batch-id "$ARCHIVE_BATCH_ID" \
  --soft-limit-gb "${ARCHIVE_SOFT_LIMIT_GB:-180}" \
  --report "$PROJECT_ROOT/reports/storage/archive-${SLURM_JOB_ID}.json" \
  "${arguments[@]}"
