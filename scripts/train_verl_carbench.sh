#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <ablation-config> <run-id> [--dry-run]" >&2
  exit 2
fi

config_path="$1"
run_id="$2"
shift 2

exec python -B scripts/launch_verl.py \
  --config "$config_path" \
  --run-id "$run_id" \
  "$@"
