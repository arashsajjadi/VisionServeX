#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the legacy MMRotate sidecar (mmcv-full 1.7 / mmdet 2.x / mmrotate
# 0.3.4 on torch 1.13.0+cu117). The build is intentionally separate from
# the v2.9 OpenMMLab sidecar so the two trees never collide.

set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-visionservex-mmrotate-legacy:v2.9.0}"

if ! command -v docker >/dev/null 2>&1; then
  echo "{\"code\": \"DOCKER_REQUIRED\"}"
  exit 2
fi

REPO_ROOT="$(dirname "$(dirname "$(readlink -f "$0")")")"

docker build \
  --tag "$IMAGE_TAG" \
  --file "$REPO_ROOT/docker/mmrotate-legacy/Dockerfile" \
  "$REPO_ROOT"

echo "{\"code\": \"OK\", \"image\": \"$IMAGE_TAG\"}"
