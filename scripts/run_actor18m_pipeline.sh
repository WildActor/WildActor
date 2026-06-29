#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/actor18m_pipeline.yaml}"
shift $(( $# >= 1 ? 1 : 0 ))
python -m wildactor.data.pipeline --config "$CONFIG" "$@"
