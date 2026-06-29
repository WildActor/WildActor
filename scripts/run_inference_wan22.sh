#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/inference_wan22.yaml}"
REQUEST="${2:-examples/inference_request.json}"
shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))
python -m wildactor.inference.infer_wan22 --config "$CONFIG" --request "$REQUEST" "$@"
