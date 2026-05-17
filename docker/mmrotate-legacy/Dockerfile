# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
#
# VisionServeX — MMRotate legacy sidecar (OBB / oriented detection).
#
# This image isolates the OpenMMLab 0.x stack required by mmrotate
# 0.3.4: torch 1.13, mmcv-full 1.7, mmdet 2.28. mmrotate's mmcv 2.x port
# was never released, so this is the only way to run Oriented R-CNN /
# RTMDet-R on a permissive base.
#
# Verified target pins (build is currently external to host CI — image
# build can take 20+ minutes and pulls ~6 GB of CUDA wheels):
#   python      3.8
#   setuptools  < 72
#   torch       1.13.1+cu116  (1.13.0-cu117 image was removed from Docker Hub)
#   torchvision 0.14.1
#   mmcv-full   1.7.2
#   mmdet       2.28.2
#   mmrotate    0.3.4
#
# Usage:
#   docker build -t visionservex-mmrotate-legacy:v3.0.0 \
#     -f docker/mmrotate-legacy/Dockerfile .
#   docker run --gpus all --rm \
#     visionservex-mmrotate-legacy:v3.0.0 \
#     openmmlab smoke-test oriented-rcnn \
#       --image examples/images/aerial.jpg --device cpu --out /tmp/out.json

# pytorch:1.13.0-cuda11.7 was removed from Docker Hub; 1.13.1-cuda11.6 is the
# closest available official image and is compatible with mmcv-full 1.7.2.
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VISIONSERVEX_CACHE_DIR=/cache/visionservex

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libjpeg-turbo8 libpng16-16 libwebp7 libtiff5 \
    git wget build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install -U pip "setuptools<72" wheel openmim

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY docs ./docs
RUN pip install --no-deps -e .
RUN pip install pydantic pydantic-settings typer rich pillow PyYAML httpx tenacity click psutil "numpy<2"

# Legacy MMRotate stack — never combine with the mmcv 2.x sidecar image.
RUN mim install "mmcv-full==1.7.2" \
        -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html && \
    mim install "mmdet==2.28.2" && \
    pip install "mmrotate==0.3.4"

RUN python -c "import mmrotate; print('mmrotate', mmrotate.__version__)"

VOLUME ["/cache/visionservex"]
ENTRYPOINT ["visionservex"]
CMD ["openmmlab", "doctor"]
