# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
#
# Florence-2 sidecar — isolated Python 3.11 + transformers<5 environment.
#
# Florence-2's remote code targets the transformers 4.x surface and needs a
# `tokenizers` build that has NO Python-3.13 wheel, so it cannot run in the main
# VisionServeX env (py3.13 / transformers 5.x). This image pins:
#   python      3.11
#   torch       CPU (no CUDA — Florence-2 inference is light)
#   transformers 4.44.2 (Florence-2-known-good; has _supports_sdpa + legacy cache)
#   tokenizers  the matching prebuilt wheel (exists for py3.11)
#
# Build:
#   docker build -t visionservex-florence2:v321 -f docker/florence2/Dockerfile docker/florence2
# Run (mount your HF cache so weights are not re-downloaded; MIT-licensed):
#   docker run --rm -p 127.0.0.1:8091:8091 \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface visionservex-florence2:v321

FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# OpenCV/torch runtime libs (timm/Pillow image ops).
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image small; Florence-2 inference is light.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.2.2"
RUN pip install \
    "transformers==4.44.2" \
    "tokenizers>=0.19,<0.20" \
    "einops>=0.7" \
    "timm>=0.9" \
    "pillow>=10.0" \
    "fastapi>=0.110" \
    "uvicorn>=0.27" \
    "python-multipart>=0.0.9" \
    "accelerate>=0.28"

WORKDIR /app
COPY server.py ./server.py

EXPOSE 8091
# Sanity probe: Florence-2's processing module must import under this stack.
RUN python -c "import torch, transformers, tokenizers, einops, timm; print('florence2 stack', torch.__version__, transformers.__version__, tokenizers.__version__)"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8091"]
