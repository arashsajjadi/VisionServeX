#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Real RTMPose-m + RTMDet-tiny OpenMMLab smoke. Creates an isolated conda
# env on Python 3.10 with setuptools<72, installs the pinned OpenMMLab
# stack, then runs `visionservex openmmlab smoke-test`.
#
# Verified on the release host with these pinned versions (2026-05-16):
#   torch 2.1.0+cu121, mmcv 2.1.0, mmengine 0.10.7, mmpose 1.3.2, mmdet 3.3.0
#   numpy 1.26.4, xtcocotools rebuilt from source against numpy 1.26
#
# Usage:
#   bash scripts/run_openmmlab_rtmpose_smoke.sh [env_name] [image_path]
#
# Exits with:
#   0 — RTMPose-m and RTMDet-tiny both ran and wrote JSON.
#   2 — CONDA_REQUIRED
#   3 — INPUT_NOT_FOUND
#   4 — OPENMMLAB_API_UNSUPPORTED (inference call failed)
#
# WHY THESE PINS:
# * mmcv 2.2.0 source build needs pkg_resources (removed in setuptools>=72)
#   and mmdet 3.3.0 rejects mmcv 2.2; mmcv 2.1.0 has a prebuilt torch 2.1 wheel.
# * mmrotate 0.3.4 forces mmdet/mmcv 1.x downgrade — keep it out of this env.
# * xtcocotools wheel is built against numpy 1.x ABI; numpy must stay <2.

set -euo pipefail

ENV_NAME="${1:-vsx-openmmlab-py310}"
IMAGE="${2:-examples/images/person.jpg}"
PYVER="${PYVER:-3.10}"
TORCH_TAG="${TORCH_TAG:-cu121}"

if ! command -v conda >/dev/null 2>&1; then
  echo "{\"code\": \"CONDA_REQUIRED\", \"fix\": \"Install Miniconda from https://docs.conda.io/en/latest/miniconda.html\"}"
  exit 2
fi
if [ ! -f "$IMAGE" ]; then
  echo "{\"code\": \"INPUT_NOT_FOUND\", \"image\": \"$IMAGE\"}"
  exit 3
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" "python=$PYVER" -y -q
fi

# Pin setuptools<72 so pkg_resources stays available for source builds.
conda run -n "$ENV_NAME" python -m pip install -q -U pip "setuptools<72" wheel openmim
conda run -n "$ENV_NAME" python -m pip install -q \
  "torch==2.1.0" "torchvision==0.16.0" \
  --index-url "https://download.pytorch.org/whl/$TORCH_TAG"

conda run -n "$ENV_NAME" mim install "mmengine>=0.10"
conda run -n "$ENV_NAME" mim install "mmcv==2.1.0" \
  -f "https://download.openmmlab.com/mmcv/dist/$TORCH_TAG/torch2.1.0/index.html"
conda run -n "$ENV_NAME" python -m pip install -q --no-deps mmpose
conda run -n "$ENV_NAME" python -m pip install -q xtcocotools munkres json_tricks scipy
conda run -n "$ENV_NAME" mim install "mmdet==3.3.0"

# Force numpy<2 + rebuild xtcocotools binary against it.
conda run -n "$ENV_NAME" python -m pip install -q --force-reinstall --no-deps "numpy==1.26.4"
conda run -n "$ENV_NAME" python -m pip install -q --force-reinstall --no-binary xtcocotools \
  --no-build-isolation xtcocotools

# Re-pin setuptools<72 in case xtcocotools rebuild bumped it.
conda run -n "$ENV_NAME" python -m pip install -q --force-reinstall --no-deps "numpy==1.26.4" "setuptools<72"

# Install VisionServeX (editable) — minimum runtime deps.
conda run -n "$ENV_NAME" python -m pip install -q --no-deps \
  -e "$(dirname "$(dirname "$(readlink -f "$0")")")"
conda run -n "$ENV_NAME" python -m pip install -q \
  pydantic pydantic-settings typer rich pillow PyYAML httpx tenacity click psutil

# RTMPose-m smoke (auto-pulls rtmpose-m + rtmdet-m via pose2d='human').
conda run -n "$ENV_NAME" visionservex openmmlab smoke-test rtmpose-m \
  --image "$IMAGE" --device cpu --out /tmp/vsx_rtmpose_m.json --json

# RTMDet-tiny detection smoke.
DET_IMAGE="${3:-examples/images/street.jpg}"
if [ -f "$DET_IMAGE" ]; then
  conda run -n "$ENV_NAME" visionservex openmmlab smoke-test rtmdet-tiny-coco \
    --image "$DET_IMAGE" --device cpu --out /tmp/vsx_rtmdet_tiny.json --json
fi
