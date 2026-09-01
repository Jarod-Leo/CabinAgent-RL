#!/usr/bin/env bash
set -euo pipefail

sinfo -N -p cluster02 \
  -O "NodeList:18,StateComplete:12,CPUsState:15,Gres:28,GresUsed:28" \
  | awk 'NR == 1 || /pro6000/'
