#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <start|resume> [run-id] [target-steps]" >&2
  exit 2
fi
phase="$1"
run_id="${2:-}"
target_steps="${3:-250}"
[[ "$target_steps" == "250" ]] || {
  echo "Formal F10 now runs continuously to the fixed target step 250" >&2
  exit 2
}
exec bash scripts/submit_fallback_ablation.sh f10 "$phase" ${run_id:+"$run_id"}
