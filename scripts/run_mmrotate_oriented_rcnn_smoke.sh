#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run Oriented R-CNN inside the legacy MMRotate sidecar. Requires that
# scripts/build_mmrotate_legacy_sidecar.sh has been run first.
#
# This script emits a CERTIFIED BLOCKER if Docker or the legacy image is
# not available — never a fake smoke result.

set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-visionservex-mmrotate-legacy:v3.0.0}"
IMAGE_INPUT="${1:-examples/images/aerial.jpg}"
OUT_FILE="${2:-/tmp/vsx_oriented_rcnn.json}"

if ! command -v docker >/dev/null 2>&1; then
  echo "{\"code\": \"DOCKER_REQUIRED\", \"fix\": \"Install Docker Engine and run scripts/build_mmrotate_legacy_sidecar.sh first.\"}"
  exit 2
fi
if [ ! -f "$IMAGE_INPUT" ]; then
  echo "{\"code\": \"INPUT_NOT_FOUND\", \"image\": \"$IMAGE_INPUT\"}"
  exit 3
fi
if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  cat <<JSON
{
  "code": "MMROTATE_LEGACY_IMAGE_REQUIRED",
  "image_tag": "$IMAGE_TAG",
  "fix": "Run scripts/build_mmrotate_legacy_sidecar.sh first.",
  "future_unblock_condition": "mmrotate 1.x release compatible with mmcv 2.x — track https://github.com/open-mmlab/mmrotate/issues for the announcement."
}
JSON
  exit 4
fi

docker run --rm \
  -v "$(pwd):/work" \
  -w /work \
  "$IMAGE_TAG" openmmlab smoke-test oriented-rcnn \
    --image "$IMAGE_INPUT" --device cpu --out "$OUT_FILE" --json
