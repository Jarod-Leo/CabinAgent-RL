#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run-id> [trainer arguments]" >&2
  exit 2
fi

run_id="$1"
shift
gate_report="${DIRECT_RL_GATE_REPORT:-reports/direct_rl_gate_G02_131950.json}"
sft_config="${SFT_CONFIG:-configs/train/sft_fallback_lora.yaml}"
if [[ ! -s "$gate_report" ]]; then
  echo "Missing direct-RL gate report: $gate_report" >&2
  exit 2
fi
python -B -c 'import json,sys; value=json.load(open(sys.argv[1])); assert value.get("passed") is False, "fallback requires a failed gate"' "$gate_report"

exec python -B scripts/train_sft_fallback.py \
  --config "$sft_config" \
  --run-id "$run_id" \
  --gate-report "$gate_report" \
  "$@"
