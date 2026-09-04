#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <batch-id> <project-relative-path> [project-relative-path ...]" >&2
  exit 2
fi
batch_id="$1"
shift
PROJECT_ROOT="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/projects/cabinagentrlarchive/CabinAgent-RL}"
GPU_ENV="${GPU_ENV:-$PROJECT_ROOT/envs/cabinagentrl}"
PYTHON_BIN="${PYTHON_BIN:-$GPU_ENV/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing project Python: $PYTHON_BIN" >&2; exit 1; }
[[ -d "${ARCHIVE_ROOT%/CabinAgent-RL}" ]] || { echo "HDD project is missing: ${ARCHIVE_ROOT%/CabinAgent-RL}" >&2; exit 1; }
relative_paths="$(IFS=:; echo "$*")"
cd "$PROJECT_ROOT"
mkdir -p logs/slurm reports/storage
job_id="$(sbatch --parsable \
  --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,ARCHIVE_ROOT=$ARCHIVE_ROOT,ARCHIVE_BATCH_ID=$batch_id,ARCHIVE_RELATIVE_PATHS=$relative_paths,ARCHIVE_SOFT_LIMIT_GB=${ARCHIVE_SOFT_LIMIT_GB:-180}" \
  scripts/slurm_archive_storage.sbatch)"
[[ -n "$job_id" ]] || { echo "Empty storage archive Job ID" >&2; exit 1; }
echo "STORAGE_ARCHIVE_SUBMITTED job_id=$job_id batch_id=$batch_id"
