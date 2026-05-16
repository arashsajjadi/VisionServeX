#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the VisionServeX OpenMMLab sidecar image. Tag default is
# visionservex-openmmlab:v3.0.0; override with the IMAGE_TAG env var.
#
# Usage:
#   bash scripts/build_openmmlab_sidecar.sh
#   IMAGE_TAG=visionservex-openmmlab:dev bash scripts/build_openmmlab_sidecar.sh

set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-visionservex-openmmlab:v3.0.0}"

if ! command -v docker >/dev/null 2>&1; then
  echo "{\"code\": \"DOCKER_REQUIRED\", \"fix\": \"Install Docker Engine (https://docs.docker.com/engine/install/).\"}"
  exit 2
fi

REPO_ROOT="$(dirname "$(dirname "$(readlink -f "$0")")")"

docker build \
  --tag "$IMAGE_TAG" \
  --file "$REPO_ROOT/docker/openmmlab/Dockerfile" \
  "$REPO_ROOT"

echo "{\"code\": \"OK\", \"image\": \"$IMAGE_TAG\"}"
