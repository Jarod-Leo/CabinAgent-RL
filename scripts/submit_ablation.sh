#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {vanilla|turn_discount|lata|prm_lite|prm_lite_lata}" >&2
  exit 2
fi

experiment="$1"
case "$experiment" in
  vanilla|turn_discount|lata|prm_lite|prm_lite_lata) ;;
  *) echo "Unknown ablation: $experiment" >&2; exit 2 ;;
esac

project_root="${PROJECT_ROOT:-/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL}"
cd "$project_root"
config="configs/train/ablations/$experiment.yaml"
run_id="${RUN_ID:-${experiment}_$(date -u +%Y%m%dT%H%M%SZ)}"

python -B scripts/init_experiment.py --config "$config" --run-id "$run_id"
sbatch \
  --job-name="car-${experiment}" \
  --export="ALL,PROJECT_ROOT=$project_root,EXPERIMENT_CONFIG=$config,RUN_ID=$run_id" \
  scripts/slurm_dual_pro6000.sbatch
