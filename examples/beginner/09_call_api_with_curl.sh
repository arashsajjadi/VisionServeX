#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
# Beginner example 09 — call the local HTTP API with curl.
#
# Start the server first:
#   visionservex serve
# (in another terminal)
set -euo pipefail

HOST="${VSX_HOST:-http://127.0.0.1:8080}"
IMG="${1:-examples/images/street.jpg}"

if [[ -n "${VISIONSERVEX_AUTH__API_KEY:-}" ]]; then
  AUTH=(-H "Authorization: Bearer ${VISIONSERVEX_AUTH__API_KEY}")
else
  AUTH=()
fi

echo "--- /health ---"
curl -fsS "${AUTH[@]}" "$HOST/health" | jq .

echo "--- /devices ---"
curl -fsS "${AUTH[@]}" "$HOST/devices" | jq .

echo "--- /models (first 3) ---"
curl -fsS "${AUTH[@]}" "$HOST/models" | jq '.models[0:3]'

echo "--- /detect (mock-detect) ---"
curl -fsS "${AUTH[@]}" -F "image=@${IMG}" -F "model_id=mock-detect" "$HOST/detect" | jq .

echo
echo "Tip: enable auto-pull in config to download real models on first request:"
echo "  export VISIONSERVEX_MODELS__AUTO_PULL=true"
