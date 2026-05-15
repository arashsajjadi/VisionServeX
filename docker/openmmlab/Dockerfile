# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
#
# VisionServeX — OpenMMLab expert integration image.
#
# This image installs the full OpenMMLab toolchain via openmim so that
# RTMPose, RTMDet-R/R2, Co-DINO-Inst, and InternImage can be used without
# requiring a native mmcv/mmengine build on the host.
#
# Usage:
#   docker build -t visionservex-openmmlab:latest -f docker/openmmlab/Dockerfile .
#   docker run --gpus all --rm -p 127.0.0.1:8080:8080 visionservex-openmmlab:latest
#
# WARNING: This image is GPU-only (CUDA). CPU-only inference with these
# models is too slow to be useful. Use the standard CPU image for
# non-OpenMMLab models.
#
# For CPU-only use, use docker/Dockerfile instead.

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VISION_SERVEX_CACHE_DIR=/cache/visionservex \
    VISIONSERVEX_SERVER__HOST=0.0.0.0 \
    VISIONSERVEX_SERVER__PORT=8080

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libjpeg-turbo8 libpng16-16 libwebp7 libtiff5 \
    git wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY docs ./docs

# Install VisionServeX base + server + hf extras
RUN pip install --upgrade pip && pip install ".[server,hf,rfdetr]"

# Install OpenMMLab toolchain via openmim
RUN pip install openmim && \
    mim install mmengine mmcv "mmdet>=3.0" "mmpose>=1.0" "mmpretrain>=1.0" || \
    echo "WARNING: some openmim packages failed to install; RTMPose/RTMDet may not work"

# Install mmrotate for OBB models (may need separate CUDA build)
RUN mim install mmrotate || echo "WARNING: mmrotate installation failed; RTMDet-R/R2 not available"

# Drop privileges
RUN useradd --create-home --shell /bin/bash visionservex \
 && mkdir -p /cache/visionservex \
 && chown -R visionservex:visionservex /cache /app

USER visionservex

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health').read()" || exit 1

CMD ["visionservex", "serve"]
