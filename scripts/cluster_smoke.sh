#!/usr/bin/env bash
set -euo pipefail

duration_seconds="${SMOKE_DURATION_SECONDS:-1800}"
heartbeat_seconds="${SMOKE_HEARTBEAT_SECONDS:-60}"

if ! [[ "$duration_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "SMOKE_DURATION_SECONDS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$heartbeat_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "SMOKE_HEARTBEAT_SECONDS must be a positive integer" >&2
  exit 2
fi

started_at=$SECONDS
echo "SMOKE_START timestamp=$(date -Is) host=$(hostname) duration_seconds=$duration_seconds"
module load Miniforge3/24.11.3-1
python --version
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

python -B -m src.eval.run_baseline --benchmark all
python -B scripts/build_prm_lite_data.py --input data/eval_cache/all_trajectories.jsonl

required_outputs=(
  data/eval_cache/all_trajectories.jsonl
  data/reward/prm_lite_debug.jsonl
  reports/eval_summary.csv
  reports/failure_taxonomy.md
)

for output_path in "${required_outputs[@]}"; do
  test -s "$output_path"
done

while (( SECONDS - started_at < duration_seconds )); do
  elapsed=$((SECONDS - started_at))
  remaining=$((duration_seconds - elapsed))
  sleep_for=$heartbeat_seconds
  if (( remaining < sleep_for )); then
    sleep_for=$remaining
  fi

  echo "SMOKE_HEARTBEAT timestamp=$(date -Is) elapsed_seconds=$elapsed remaining_seconds=$remaining"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
  sleep "$sleep_for"
done

echo "SMOKE_OK timestamp=$(date -Is) elapsed_seconds=$((SECONDS - started_at))"
