# VisionServeX CPU image.
#
# This image keeps the base install light. Heavy backends (torch with CUDA,
# SAM 2, OpenMMLab) should be installed in a derivative image.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VISION_SERVEX_CACHE_DIR=/cache/visionservex \
    VISIONSERVEX_SERVER__HOST=0.0.0.0 \
    VISIONSERVEX_SERVER__PORT=8080

# OS deps. libjpeg/libpng for Pillow at runtime; build-essential for any
# downstream wheels that may need compilation.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      libjpeg62-turbo \
      libpng16-16 \
      libwebp7 \
      libtiff6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY docs ./docs

# We install with the server extra so the container is useful out of the box.
RUN pip install --upgrade pip && pip install ".[server]"

# Drop privileges.
RUN useradd --create-home --shell /bin/bash visionservex \
 && mkdir -p /cache/visionservex \
 && chown -R visionservex:visionservex /cache /app

USER visionservex

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health').read()" || exit 1

CMD ["visionservex", "serve"]
