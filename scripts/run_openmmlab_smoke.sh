#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Real OpenMMLab smoke runner. Creates an isolated conda env, installs
# mmengine / mmcv / mmpose / mmdet via openmim, then runs RTMPose-m on a
# supplied image. The env name and target model are configurable.
#
# Known blocker on hosts with Python 3.13: openmim hard-imports
# pkg_resources (removed from setuptools >=72), and mmcv 2.2.0's source
# build also imports pkg_resources at install time. Therefore this
# script targets Python 3.10 by default and pins setuptools < 72.
#
# Usage:
#   bash scripts/run_openmmlab_smoke.sh /path/to/person.jpg [env_name]
#
# Exits 0 if RTMPose-m runs; nonzero with a structured error otherwise.

set -euo pipefail

IMAGE_PATH="${1:-examples/images/person.jpg}"
ENV_NAME="${2:-visionservex-openmmlab}"
PYVER="${PYVER:-3.10}"
MODEL_ID="${MODEL_ID:-rtmpose-m}"

if ! command -v conda >/dev/null 2>&1; then
  echo "{\"code\": \"CONDA_REQUIRED\", \"message\": \"conda is required for OpenMMLab isolated env.\", \"fix\": \"Install Miniconda from https://docs.conda.io/en/latest/miniconda.html\"}"
  exit 2
fi

if [ ! -f "$IMAGE_PATH" ]; then
  echo "{\"code\": \"INPUT_NOT_FOUND\", \"image\": \"$IMAGE_PATH\"}"
  exit 3
fi

# Create the env if missing.
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[create-env] $ENV_NAME (Python $PYVER)"
  conda create -n "$ENV_NAME" "python=$PYVER" -y -q
fi

# Pin setuptools < 72 so pkg_resources stays available for mmcv source builds.
conda run -n "$ENV_NAME" python -m pip install -q -U pip "setuptools<72" wheel
conda run -n "$ENV_NAME" python -m pip install -q openmim torch torchvision

# mim install (network-heavy, ~1 GB). Each step prints what's happening.
conda run -n "$ENV_NAME" mim install "mmengine>=0.10"
conda run -n "$ENV_NAME" mim install "mmcv>=2.0.0"
conda run -n "$ENV_NAME" mim install "mmpose>=1.3"
conda run -n "$ENV_NAME" mim install "mmdet>=3.3"

# Smoke: build the inferencer and run on the supplied image.
conda run -n "$ENV_NAME" python - <<PY
from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer("$MODEL_ID")
out = next(inferencer("$IMAGE_PATH", show=False))
print({
    "model_id": "$MODEL_ID",
    "image": "$IMAGE_PATH",
    "status": "ok",
    "result_keys": list(out.keys()),
})
PY
