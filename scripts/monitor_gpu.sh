#!/usr/bin/env bash
set -euo pipefail

interval="${1:-5}"
echo "timestamp,index,uuid,name,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_draw_w"
while true; do
  timestamp="$(date --iso-8601=seconds)"
  nvidia-smi \
    --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits | sed "s/^/${timestamp},/"
  sleep "$interval"
done
