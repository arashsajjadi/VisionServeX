#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Thin wrapper that runs ONLY the RTMDet-tiny COCO smoke inside the
# Python 3.10 conda sidecar provisioned by run_openmmlab_rtmpose_smoke.sh.
# Reuses the same env so we don't pay the install cost twice.
#
# Usage:
#   bash scripts/run_openmmlab_rtmdet_smoke.sh [env_name] [image_path]

set -euo pipefail

ENV_NAME="${1:-vsx-openmmlab-py310}"
IMAGE="${2:-examples/images/street.jpg}"

if ! command -v conda >/dev/null 2>&1; then
  echo "{\"code\": \"CONDA_REQUIRED\"}"
  exit 2
fi
if [ ! -f "$IMAGE" ]; then
  echo "{\"code\": \"INPUT_NOT_FOUND\", \"image\": \"$IMAGE\"}"
  exit 3
fi
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "{\"code\": \"ENV_NOT_FOUND\", \"env\": \"$ENV_NAME\", \"fix\": \"Run scripts/run_openmmlab_rtmpose_smoke.sh first.\"}"
  exit 4
fi

conda run -n "$ENV_NAME" visionservex openmmlab smoke-test rtmdet-tiny-coco \
  --image "$IMAGE" --device cpu --out /tmp/vsx_rtmdet_tiny.json --json
