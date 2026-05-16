#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run RTMPose-m and RTMDet-tiny smoke inside the prebuilt OpenMMLab Docker
# sidecar. CPU device by default (the RTX 5080 sm_120 architecture isn't
# supported by torch 2.1.0 — see scripts/run_openmmlab_rtmpose_smoke.sh).
#
# Usage:
#   bash scripts/run_openmmlab_sidecar_smoke.sh
#   IMAGE_TAG=visionservex-openmmlab:v2.9.0 \
#     POSE_IMAGE=examples/images/person.jpg \
#     DET_IMAGE=examples/images/street.jpg \
#     bash scripts/run_openmmlab_sidecar_smoke.sh

set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-visionservex-openmmlab:v2.9.0}"
POSE_IMAGE="${POSE_IMAGE:-examples/images/person.jpg}"
DET_IMAGE="${DET_IMAGE:-examples/images/street.jpg}"
OUT_DIR="${OUT_DIR:-/tmp/vsx-openmmlab-sidecar}"

if ! command -v docker >/dev/null 2>&1; then
  echo "{\"code\": \"DOCKER_REQUIRED\"}"
  exit 2
fi
if [ ! -f "$POSE_IMAGE" ]; then
  echo "{\"code\": \"INPUT_NOT_FOUND\", \"image\": \"$POSE_IMAGE\"}"
  exit 3
fi
mkdir -p "$OUT_DIR"

# RTMPose-m
docker run --rm \
  -v "$(pwd):/work" \
  -v "$OUT_DIR:/out" \
  -w /work \
  "$IMAGE_TAG" openmmlab smoke-test rtmpose-m \
    --image "$POSE_IMAGE" --device cpu --out /out/rtmpose_m.json --json

# RTMDet-tiny (only if the detection image is present)
if [ -f "$DET_IMAGE" ]; then
  docker run --rm \
    -v "$(pwd):/work" \
    -v "$OUT_DIR:/out" \
    -w /work \
    "$IMAGE_TAG" openmmlab smoke-test rtmdet-tiny-coco \
      --image "$DET_IMAGE" --device cpu --out /out/rtmdet_tiny.json --json
fi

echo "{\"code\": \"OK\", \"pose\": \"$OUT_DIR/rtmpose_m.json\", \"detect\": \"$OUT_DIR/rtmdet_tiny.json\"}"
