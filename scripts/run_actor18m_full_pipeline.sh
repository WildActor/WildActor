#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-Actor-18M/configs/full_pipeline.yaml}"
MACHINE_ID="${MACHINE_ID:-0}"
TOTAL_MACHINES="${TOTAL_MACHINES:-1}"

python -m wildactor.data.full_pipeline \
  --config "$CONFIG" \
  --machine_id "$MACHINE_ID" \
  --total_machines "$TOTAL_MACHINES" \
  "$@"
