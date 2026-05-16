#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Real MaskDINO sidecar runner. Creates a Detectron2 conda env, clones
# MaskDINO, then runs the official demo on the supplied image and
# checkpoint. We do not invent checkpoint URLs — the caller must supply
# CKPT (a local path to a MaskDINO weights file obtained from the
# official MaskDINO README's model-zoo table).
#
# Usage:
#   CKPT=/path/to/maskdino_swinl_50ep.pth \
#     bash scripts/run_maskdino_smoke.sh /path/to/street.jpg [env_name] [config]
#
# Structured exit codes:
#   2 CONDA_REQUIRED
#   3 CHECKPOINT_REQUIRED / INPUT_NOT_FOUND
#   4 DETECTRON2_REQUIRED / CUDA_OPS_REQUIRED
#
# This script never downloads weights silently. If CKPT is missing it
# exits with CHECKPOINT_REQUIRED and the user-facing pointer to the
# upstream README.

set -euo pipefail

IMAGE_PATH="${1:-examples/images/street.jpg}"
ENV_NAME="${2:-visionservex-maskdino}"
CONFIG_PATH="${3:-configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_4s.yaml}"
PYVER="${PYVER:-3.10}"
CUDA_TAG="${CUDA_TAG:-cu121}"
REPO_DIR="${REPO_DIR:-/tmp/MaskDINO}"

if ! command -v conda >/dev/null 2>&1; then
  echo "{\"code\": \"CONDA_REQUIRED\"}"
  exit 2
fi

if [ ! -f "$IMAGE_PATH" ]; then
  echo "{\"code\": \"INPUT_NOT_FOUND\", \"image\": \"$IMAGE_PATH\"}"
  exit 3
fi

if [ -z "${CKPT:-}" ] || [ ! -f "$CKPT" ]; then
  cat <<'JSON'
{
  "code": "CHECKPOINT_REQUIRED",
  "message": "MaskDINO checkpoint not found. Set CKPT to a local .pth path.",
  "source": "https://github.com/IDEA-Research/MaskDINO/blob/main/README.md",
  "fix": "Download a checkpoint from the upstream model zoo and re-run with CKPT=/path/to/weights.pth."
}
JSON
  exit 3
fi

# Create env if missing.
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" "python=$PYVER" -y -q
fi
conda run -n "$ENV_NAME" python -m pip install -q -U pip "setuptools<72" wheel
conda run -n "$ENV_NAME" python -m pip install -q "torch==2.2.*" "torchvision==0.17.*" \
  --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

# Detectron2 needs the matching torch/cuda. Fail fast if install errors.
conda run -n "$ENV_NAME" python -m pip install -q "detectron2 @ git+https://github.com/facebookresearch/detectron2.git" || {
  echo "{\"code\": \"DETECTRON2_REQUIRED\", \"fix\": \"See https://detectron2.readthedocs.io/en/latest/tutorials/install.html for the exact CUDA/torch pin matching your host.\"}"
  exit 4
}

# Clone MaskDINO into REPO_DIR (or refresh).
if [ ! -d "$REPO_DIR" ]; then
  git clone --depth 1 https://github.com/IDEA-Research/MaskDINO "$REPO_DIR"
fi
conda run -n "$ENV_NAME" python -m pip install -q -r "$REPO_DIR/requirements.txt"

# Run the official demo.
conda run -n "$ENV_NAME" python "$REPO_DIR/demo/demo.py" \
  --config-file "$REPO_DIR/$CONFIG_PATH" \
  --input "$IMAGE_PATH" \
  --output /tmp/maskdino_smoke.png \
  --opts MODEL.WEIGHTS "$CKPT" || {
  echo "{\"code\": \"CUDA_OPS_REQUIRED\", \"fix\": \"Detectron2's custom CUDA ops are not loadable; align CUDA toolkit and torch wheel.\"}"
  exit 4
}

echo "{\"code\": \"OK\", \"output\": \"/tmp/maskdino_smoke.png\", \"config\": \"$CONFIG_PATH\"}"
