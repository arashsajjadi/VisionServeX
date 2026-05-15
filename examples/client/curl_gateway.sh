#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# VisionServeX Local Gateway — curl examples
# Start server: visionservex gateway start

set -euo pipefail
HOST="${VSX_HOST:-http://127.0.0.1:8080}"
IMG="${1:-examples/images/street.jpg}"

echo "--- /health ---"
curl -fsS "$HOST/health" | jq .

echo "--- /gateway/status ---"
curl -fsS "$HOST/gateway/status" | jq '{device: .best_device.name, loaded: (.loaded_models|length), jobs: .active_jobs}'

echo "--- /models (first 5) ---"
curl -fsS "$HOST/models" | jq '[.models[:5][] | {id, task, implementation_status}]'

echo "--- /detect ---"
curl -fsS -F "image=@${IMG}" -F "model_id=dfine-n" "$HOST/detect" | jq '{device, latency_ms, n_results: (.results|length)}'

echo "--- /classify ---"
curl -fsS -F "image=@${IMG}" -F "model_id=swinv2-tiny" "$HOST/classify" | jq '{device, latency_ms, top3: .results[:3]}'
