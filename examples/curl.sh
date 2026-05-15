#!/usr/bin/env bash
# VisionServeX HTTP API examples.
#
# Start the server first:
#   visionservex serve
#
# Optionally enable auth:
#   export VISIONSERVEX_AUTH__ENABLED=true
#   export VISIONSERVEX_AUTH__API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")

set -euo pipefail

HOST="${VSX_HOST:-http://127.0.0.1:8080}"
IMG="${1:-image.jpg}"
AUTH=()

if [[ -n "${VISIONSERVEX_AUTH__API_KEY:-}" ]]; then
  AUTH=(-H "Authorization: Bearer ${VISIONSERVEX_AUTH__API_KEY}")
fi

echo "--- /health ---"
curl -fsS "${AUTH[@]}" "$HOST/health" | jq .

echo "--- /models (first 3) ---"
curl -fsS "${AUTH[@]}" "$HOST/models" | jq '.models | .[0:3]'

echo "--- /detect (mock-detect) ---"
curl -fsS "${AUTH[@]}" -F "image=@${IMG}" -F "model_id=mock-detect" "$HOST/detect" | jq .

echo "--- /metrics ---"
curl -fsS "${AUTH[@]}" "$HOST/metrics" | jq .
