#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
set -euo pipefail
HOST="${VSX_HOST:-http://127.0.0.1:8080}"
IMG="${1:-examples/images/street.jpg}"
MODEL="${2:-mock-detect}"
curl -fsS -F "image=@${IMG}" -F "model_id=${MODEL}" "${HOST}/detect" | jq .
