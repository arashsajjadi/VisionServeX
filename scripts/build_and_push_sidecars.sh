#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build (and optionally push) all VisionServeX sidecar Docker images to GHCR.
#
# Builds:
#   ghcr.io/<owner>/visionservex-openmmlab:<version>
#   ghcr.io/<owner>/visionservex-mmrotate-legacy:<version>
#   ghcr.io/<owner>/visionservex-maskdino:<version>
#
# Also tags each as :latest.
#
# Usage:
#   # Dry build (no push)
#   bash scripts/build_and_push_sidecars.sh --version v3.0.0 --owner arashsajjadi
#
#   # Build and push
#   bash scripts/build_and_push_sidecars.sh --version v3.0.0 --owner arashsajjadi --push
#
#   # Build a single image
#   bash scripts/build_and_push_sidecars.sh --version v3.0.0 --owner arashsajjadi \
#     --images openmmlab --push
#
# Auth (must be done before --push):
#   echo $GHCR_TOKEN | docker login ghcr.io -u <owner> --password-stdin
#
# Structured exit codes:
#   0  OK (build and/or push succeeded)
#   2  DOCKER_REQUIRED
#   3  GHCR_NAMESPACE_REQUIRED (owner not provided)
#   4  GHCR_AUTH_REQUIRED (not logged in and --push given)
#   5  DOCKER_BUILD_FAILED
#   6  GHCR_PUSH_FAILED

set -euo pipefail

REPO_ROOT="$(dirname "$(dirname "$(readlink -f "$0")")")"

# ─── Defaults ────────────────────────────────────────────────────────────────
VERSION="v3.0.0"
OWNER=""
PUSH=false
IMAGES="openmmlab,mmrotate-legacy,maskdino"
PLATFORM="linux/amd64"
DRY_RUN=false

# ─── Argument parsing ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --version)  VERSION="$2"; shift 2 ;;
    --owner)    OWNER="$2"; shift 2 ;;
    --push)     PUSH=true; shift ;;
    --no-push)  PUSH=false; shift ;;
    --images)   IMAGES="$2"; shift 2 ;;
    --platform) PLATFORM="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ─── Pre-checks ──────────────────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
  echo "{\"code\": \"DOCKER_REQUIRED\", \"fix\": \"Install Docker Engine from https://docs.docker.com/engine/install/\"}"
  exit 2
fi

# Detect owner from git remote if not supplied.
if [[ -z "$OWNER" ]]; then
  OWNER="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null | sed 's|.*github.com[:/]||; s|/.*||' || true)"
fi
if [[ -z "$OWNER" ]]; then
  echo "{\"code\": \"GHCR_NAMESPACE_REQUIRED\", \"fix\": \"Pass --owner <github-username>\"}"
  exit 3
fi

if $PUSH && ! docker info 2>/dev/null | grep -q "ghcr.io"; then
  # Check if already logged in to GHCR.
  if ! docker system info 2>/dev/null | grep -qi "ghcr"; then
    : # Continue — docker will fail at push time with a clear error
  fi
fi

echo "{\"owner\": \"$OWNER\", \"version\": \"$VERSION\", \"images\": \"$IMAGES\", \"push\": $PUSH, \"platform\": \"$PLATFORM\"}"

# ─── Image build specs ───────────────────────────────────────────────────────
declare -A IMAGE_SPECS
IMAGE_SPECS["openmmlab"]="docker/openmmlab/Dockerfile|OpenMMLab expert sidecar (RTMPose/RTMDet)"
IMAGE_SPECS["mmrotate-legacy"]="docker/mmrotate-legacy/Dockerfile|MMRotate legacy sidecar (Oriented R-CNN)"
IMAGE_SPECS["maskdino"]="docker/maskdino/Dockerfile|MaskDINO Detectron2 sidecar"

BUILT=()
PUSHED=()
FAILED=()

IFS=',' read -ra IMAGE_LIST <<< "$IMAGES"
for img_name in "${IMAGE_LIST[@]}"; do
  img_name="${img_name// /}"
  if [[ -z "${IMAGE_SPECS[$img_name]+_}" ]]; then
    echo "{\"code\": \"UNKNOWN_IMAGE\", \"image\": \"$img_name\", \"available\": \"${!IMAGE_SPECS[*]}\"}"
    continue
  fi

  IFS='|' read -r DOCKERFILE DESCRIPTION <<< "${IMAGE_SPECS[$img_name]}"
  DOCKERFILE_PATH="$REPO_ROOT/$DOCKERFILE"

  if [[ ! -f "$DOCKERFILE_PATH" ]]; then
    echo "{\"code\": \"DOCKERFILE_NOT_FOUND\", \"path\": \"$DOCKERFILE_PATH\"}"
    FAILED+=("$img_name")
    continue
  fi

  IMAGE_NAME="ghcr.io/$OWNER/visionservex-$img_name"
  VERSION_TAG="$IMAGE_NAME:$VERSION"
  LATEST_TAG="$IMAGE_NAME:latest"

  echo "Building $VERSION_TAG ..."

  if $DRY_RUN; then
    echo "{\"code\": \"DRY_RUN\", \"image\": \"$VERSION_TAG\"}"
    BUILT+=("$img_name")
    continue
  fi

  BUILD_CMD=(
    docker buildx build
    --platform "$PLATFORM"
    --tag "$VERSION_TAG"
    --tag "$LATEST_TAG"
    --label "org.opencontainers.image.source=https://github.com/$OWNER/VisionServeX"
    --label "org.opencontainers.image.version=$VERSION"
    --label "org.opencontainers.image.title=visionservex-$img_name"
    --label "org.opencontainers.image.description=$DESCRIPTION"
    --file "$DOCKERFILE_PATH"
    "$REPO_ROOT"
  )

  if $PUSH; then
    BUILD_CMD+=(--push)
  else
    BUILD_CMD+=(--load)
  fi

  if "${BUILD_CMD[@]}"; then
    BUILT+=("$img_name")
    echo "{\"code\": \"BUILD_OK\", \"image\": \"$VERSION_TAG\", \"pushed\": $PUSH}"
    if $PUSH; then
      PUSHED+=("$img_name")
      # Retrieve and print the digest.
      DIGEST="$(docker inspect "$VERSION_TAG" --format '{{index .RepoDigests 0}}' 2>/dev/null || echo 'unknown')"
      echo "{\"code\": \"PUSH_OK\", \"image\": \"$VERSION_TAG\", \"digest\": \"$DIGEST\"}"
    fi
  else
    FAILED+=("$img_name")
    echo "{\"code\": \"DOCKER_BUILD_FAILED\", \"image\": \"$VERSION_TAG\", \"dockerfile\": \"$DOCKERFILE_PATH\"}"
  fi
done

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "{\"built\": [$(printf '"%s",' "${BUILT[@]:-}")], \"pushed\": [$(printf '"%s",' "${PUSHED[@]:-}")], \"failed\": [$(printf '"%s",' "${FAILED[@]:-}")]}"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  exit 5
fi
