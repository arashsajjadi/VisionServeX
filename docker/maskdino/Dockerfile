# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
#
# VisionServeX — MaskDINO + Detectron2 expert sidecar (v3.0.0).
#
# MaskDINO is Apache-2.0 (https://github.com/IDEA-Research/MaskDINO) but
# requires Detectron2's custom CUDA ops, which cannot be bundled in the
# permissive default VisionServeX wheel. This image builds Detectron2
# from source against a specific CUDA/PyTorch combination.
#
# Checkpoint URLs (Apache-2.0, IDEA Research via GitHub releases):
#   Instance R50:    https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth
#   Instance Swin-L: https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_mask52.1ap_box58.3ap.pth
#
# Usage:
#   docker build -t ghcr.io/<owner>/visionservex-maskdino:v3.0.0 \
#     -f docker/maskdino/Dockerfile .
#   CKPT=/path/to/maskdino_*.pth \
#   docker run --gpus all --rm \
#     -v ~/.cache/visionservex:/cache/visionservex \
#     -e CKPT=/cache/visionservex/maskdino/maskdino_swinl_50ep.pth \
#     ghcr.io/<owner>/visionservex-maskdino:v3.0.0 \
#     maskdino validate maskdino-swinl-coco --json

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VISIONSERVEX_CACHE_DIR=/cache/visionservex

# OS deps including build tools needed for Detectron2 custom ops.
RUN bash -c "apt-get update -qq || (sleep 5 && apt-get update -qq)" && apt-get install -y --no-install-recommends \
    ca-certificates git wget build-essential ninja-build \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install -U pip "setuptools<72" wheel

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY docs ./docs
RUN pip install --no-deps -e .
RUN pip install pydantic pydantic-settings typer rich pillow PyYAML httpx tenacity click psutil

# Detectron2 from source — required for the custom CUDA ops.
# torch must be installed before building Detectron2 because its setup.py imports torch.
# The base image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel already provides torch 2.0.1.
RUN pip install "fvcore>=0.1.5" "iopath>=0.1.9" opencv-python-headless pycocotools
RUN pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"

# Clone MaskDINO at a pinned commit and install requirements.
RUN git clone --depth 1 https://github.com/IDEA-Research/MaskDINO /opt/MaskDINO
RUN pip install -r /opt/MaskDINO/requirements.txt || true

ENV MASKDINO_ROOT=/opt/MaskDINO

VOLUME ["/cache/visionservex"]

ENTRYPOINT ["visionservex"]
CMD ["maskdino", "doctor"]
