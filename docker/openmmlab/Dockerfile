# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
#
# VisionServeX — OpenMMLab expert integration image (v3.0.0).
#
# Pinned to the recipe that was real-smoke verified on the v2.8 release
# host (RTMPose-m + RTMDet-tiny). The mmcv 2.x ecosystem refuses to
# install on setuptools >= 72 because pkg_resources was removed, so the
# image pins setuptools < 72 and installs mmcv 2.1.0 from the prebuilt
# OpenMMLab torch 2.1.0+cu121 wheel index.
#
# Verified pins (do NOT bump without re-running the sidecar smoke):
#   python      3.10
#   setuptools  < 72
#   torch       2.1.0+cu121
#   torchvision 0.16.0
#   mmcv        2.1.0
#   mmengine    0.10.7
#   mmpose      1.3.2 (--no-deps; skips chumpy)
#   mmdet       3.3.0
#   numpy       1.26.4
#
# Usage:
#   docker build -t visionservex-openmmlab:v3.0.0 -f docker/openmmlab/Dockerfile .
#   docker run --gpus all --rm -v ~/.cache/visionservex:/cache/visionservex \
#     visionservex-openmmlab:v3.0.0 \
#     visionservex openmmlab smoke-test rtmpose-m --image examples/images/person.jpg --device cpu --out /tmp/out.json

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VISIONSERVEX_CACHE_DIR=/cache/visionservex

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libjpeg-turbo8 libpng16-16 libwebp7 libtiff5 \
    git wget build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin setuptools < 72 BEFORE anything else so pkg_resources stays available
# for mmcv / openmim source-builds.
RUN python -m pip install -U pip "setuptools<72" wheel openmim

# Project metadata + source.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY docs ./docs

# Install VisionServeX runtime deps only — no build extras (skip hf/rfdetr
# heavy stacks; those have their own images).
RUN pip install --no-deps -e .
RUN pip install \
    pydantic pydantic-settings typer rich pillow PyYAML httpx tenacity \
    click psutil "numpy==1.26.4"

# OpenMMLab pinned stack.
RUN mim install "mmengine==0.10.7" && \
    mim install "mmcv==2.1.0" \
        -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html && \
    pip install --no-deps mmpose==1.3.2 && \
    pip install xtcocotools munkres json_tricks scipy && \
    mim install "mmdet==3.3.0"

# Force numpy<2 + rebuild xtcocotools binary against numpy 1.26 ABI.
RUN pip install --force-reinstall --no-deps "numpy==1.26.4" && \
    pip install --force-reinstall --no-binary xtcocotools --no-build-isolation xtcocotools && \
    pip install --force-reinstall --no-deps "numpy==1.26.4" "setuptools<72"

# Sanity probe.
RUN python -c "import torch, mmcv, mmpose, mmdet; print('torch', torch.__version__, 'mmcv', mmcv.__version__, 'mmpose', mmpose.__version__, 'mmdet', mmdet.__version__)"

VOLUME ["/cache/visionservex"]

ENTRYPOINT ["visionservex"]
CMD ["openmmlab", "doctor"]
