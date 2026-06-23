#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
#
# Reproducible END STATE A attempt for MedSAM2 real runtime, in an ISOLATED conda
# env (python 3.12 + torch 2.5.1 CPU). NOTHING here touches the VisionServeX core
# env or installs heavy deps into it. No checkpoints are written into the repo.
#
# Upstream facts (github.com/bowang-lab/MedSAM2):
#   python 3.12, torch==2.5.1, install the MedSAM2 fork (provides `sam2`),
#   checkpoints on HF wanglab/MedSAM2 (research/education only — NON-COMMERCIAL).
#
# CPU is used on purpose: the host GPU is Blackwell (RTX 5080, sm_120) and torch
# cu124 wheels lack sm_120 kernels, so a CPU smoke is the clean, deterministic path.
#
# Usage:  bash scripts/medsam2/setup_isolated_env.sh
# Output: staged STAGE lines + a final MEDSAM2_SMOKE_JSON line from the smoke.
set -uo pipefail

ENV_NAME="vsx-medsam2"
SRC_DIR="${HOME}/.cache/vsx_medsam2_src"
CKPT_DIR="${HOME}/.cache/vsx_medsam2_ckpt"
OUT_DIR="/tmp/medsam2_out"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE="${REPO_DIR}/scripts/medsam2/real_runtime_smoke.py"

stage() { echo "STAGE $1 :: $2"; }

# Skip the CUDA extension build (would compile against the host CUDA 13 toolkit).
export SAM2_BUILD_CUDA=0
export SAM2_BUILD_ALLOW_ERRORS=1

stage env START "conda create -n ${ENV_NAME} python=3.12"
if conda env list | grep -qE "^${ENV_NAME}\s"; then
  stage env OK "env ${ENV_NAME} already exists; reusing"
else
  conda create -y -n "${ENV_NAME}" python=3.12 >/dev/null 2>&1 \
    && stage env OK "created" || { stage env FAIL "conda create failed"; exit 11; }
fi

stage torch START "pip install torch==2.5.1 torchvision==0.20.1 (cpu)"
conda run -n "${ENV_NAME}" pip install -q \
  torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
  && stage torch OK "installed" || { stage torch FAIL "torch install failed"; exit 12; }

stage clone START "git clone --depth 1 bowang-lab/MedSAM2 -> ${SRC_DIR}"
if [ -d "${SRC_DIR}/.git" ]; then
  stage clone OK "already cloned"
else
  git clone --depth 1 https://github.com/bowang-lab/MedSAM2.git "${SRC_DIR}" >/dev/null 2>&1 \
    && stage clone OK "cloned" || { stage clone FAIL "git clone failed"; exit 13; }
fi

stage sam2 START "pip install -e MedSAM2 fork (provides sam2) + huggingface_hub"
conda run -n "${ENV_NAME}" pip install -q huggingface_hub >/dev/null 2>&1
( cd "${SRC_DIR}" && conda run -n "${ENV_NAME}" pip install -q -e . ) \
  && stage sam2 OK "installed" || { stage sam2 FAIL "pip install -e MedSAM2 failed"; exit 14; }

stage import START "verify sam2 import"
conda run -n "${ENV_NAME}" python -c "import sam2, sam2.build_sam, sam2.sam2_image_predictor; print('sam2 ok')" \
  && stage import OK "sam2 importable" || { stage import FAIL "sam2 import failed"; exit 15; }

stage ckpt START "hf_hub_download wanglab/MedSAM2 MedSAM2_latest.pt"
mkdir -p "${CKPT_DIR}"
conda run -n "${ENV_NAME}" python - "$CKPT_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download
p = hf_hub_download("wanglab/MedSAM2", "MedSAM2_latest.pt", local_dir=sys.argv[1])
print("CKPT_PATH", p)
PY
RC=$?
CKPT="$(ls -1 "${CKPT_DIR}"/MedSAM2_latest.pt 2>/dev/null || true)"
if [ "${RC}" -ne 0 ] || [ -z "${CKPT}" ]; then
  stage ckpt FAIL "checkpoint download failed"; exit 16
fi
stage ckpt OK "${CKPT} ($(du -h "${CKPT}" | cut -f1))"

stage smoke START "real 2D CPU inference"
conda run -n "${ENV_NAME}" python "${SMOKE}" \
  --checkpoint "${CKPT}" --device cpu --out "${OUT_DIR}"
stage smoke DONE "exit=$?"
