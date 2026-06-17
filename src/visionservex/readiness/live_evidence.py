# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Live-verification evidence (v3.18).

A model is only allowed to report a ``*_READY_LIVE`` readiness state when a real
inference (or full train-lifecycle) smoke test *passed in this sprint* and the
evidence is committed under ``docs/qa/v318_full_model_truth/``. The committed
JSON matrices are the **evidence**; the frozensets below are the **conclusions**
baked into the package so that ``model_capabilities()`` stays fast and
weight-free at runtime (it must never load a model just to answer a question).

The two are kept in lock-step by ``tests/test_v318_capability_truth_contract.py``
and ``tools/qa/v318_sync_live_evidence.py``, which regenerate these frozensets
from the matrices and fail CI if they drift.

Honest default: both sets start EMPTY. A model is added here ONLY after its row
in the corresponding matrix has ``status == "PASS"`` and ``live_verified == true``.
Anything not listed here is reported as ``*_DERIVED_NEEDS_LIVE_CONFIRMATION`` —
capability-derived, not yet live-confirmed — never as ``*_LIVE``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# BAKED CONCLUSIONS — regenerated from the committed v3.18 matrices.
# Do not hand-edit: run ``python tools/qa/v318_sync_live_evidence.py`` after a
# live matrix run, which rewrites the two literals below from the PASS rows.
# --------------------------------------------------------------------------- #
# >>> BEGIN GENERATED: LIVE_INFERENCE_VERIFIED
LIVE_INFERENCE_VERIFIED: frozenset[str] = frozenset(
    {
        "clip-vit-base-patch32",
        "clip-vit-large-patch14",
        "convnextv2-base",
        "convnextv2-large",
        "convnextv2-tiny",
        "dfine-l",
        "dfine-l-coco",
        "dfine-l-o365-coco",
        "dfine-m",
        "dfine-m-coco",
        "dfine-m-o365-coco",
        "dfine-n",
        "dfine-n-coco",
        "dfine-s",
        "dfine-s-coco",
        "dfine-s-o365-coco",
        "dfine-x",
        "dfine-x-coco",
        "dfine-x-o365-coco",
        "dinov2-base",
        "dinov2-giant",
        "dinov2-large",
        "dinov2-small",
        "efficientsam",
        "grounded-sam",
        "grounded-sam2",
        "grounding-dino-original-swin-b",
        "grounding-dino-original-swin-t",
        "grounding-dino-swin-b",
        "grounding-dino-swin-t",
        "grounding-dino-tiny",
        "hq-sam",
        "libreyolo-dfine-l",
        "libreyolo-dfine-m",
        "libreyolo-dfine-n",
        "libreyolo-dfine-s",
        "libreyolo-dfine-x",
        "libreyolo-rtdetr-r101",
        "libreyolo-rtdetr-r50",
        "libreyolo-yolov9-c",
        "libreyolo-yolov9-m",
        "libreyolo-yolov9-s",
        "libreyolo-yolox-l",
        "libreyolo-yolox-m",
        "libreyolo-yolox-s",
        "libreyolo-yolox-x",
        "medsam",
        "mobilesam",
        "mock-classify",
        "mock-detect",
        "mock-foundation-segment",
        "mock-grounded-segment",
        "mock-obb",
        "mock-open-vocab",
        "mock-pose",
        "mock-segment",
        "oneformer-swin-large",
        "owlv2-base-patch16",
        "owlv2-large-patch14",
        "owlvit-base-patch32",
        "owlvit-large-patch14",
        "rfdetr-base",
        "rfdetr-large",
        "rfdetr-medium",
        "rfdetr-nano",
        "rfdetr-seg-medium",
        "rfdetr-seg-nano",
        "rfdetr-seg-small",
        "rfdetr-small",
        "sam-vit-base",
        "sam-vit-huge",
        "sam-vit-large",
        "sam2-hiera-base-plus",
        "sam2-hiera-large",
        "sam2-hiera-small",
        "sam2-hiera-tiny",
        "sam2.1-hiera-base-plus",
        "sam2.1-hiera-large",
        "sam2.1-hiera-small",
        "sam2.1-hiera-tiny",
        "siglip-base-patch16-224",
        "siglip2-base-patch16-224",
        "siglip2-large-patch16-256",
        "siglip2-so400m-patch14-384",
        "swinv2-base",
        "swinv2-large",
        "swinv2-small",
        "swinv2-tiny",
        "torchvision-alexnet",
        "torchvision-convnext-tiny",
        "torchvision-densenet121",
        "torchvision-efficientnet-b0",
        "torchvision-mobilenet-v2",
        "torchvision-mobilenet-v3-large",
        "torchvision-resnet101",
        "torchvision-resnet152",
        "torchvision-resnet18",
        "torchvision-resnet34",
        "torchvision-resnet50",
        "torchvision-resnext50-32x4d",
        "torchvision-wide-resnet50-2",
    }
)
# <<< END GENERATED: LIVE_INFERENCE_VERIFIED

# >>> BEGIN GENERATED: LIVE_TRAIN_VERIFIED
LIVE_TRAIN_VERIFIED: frozenset[str] = frozenset(
    {
        "libreyolo-rtdetr-r50",
        "libreyolo-yolov9-s",
        "libreyolo-yolox-s",
        "torchvision-alexnet",
        "torchvision-convnext-tiny",
        "torchvision-densenet121",
        "torchvision-efficientnet-b0",
        "torchvision-mobilenet-v2",
        "torchvision-mobilenet-v3-large",
        "torchvision-resnet101",
        "torchvision-resnet152",
        "torchvision-resnet18",
        "torchvision-resnet34",
        "torchvision-resnet50",
        "torchvision-resnext50-32x4d",
        "torchvision-wide-resnet50-2",
    }
)
# <<< END GENERATED: LIVE_TRAIN_VERIFIED

# Models that were live-tested for inference and FAILED / were blocked, mapped to
# the precise readiness state that the failure implies (DEPENDENCY_MISSING,
# WEIGHTS_MISSING, UPSTREAM_CRASH, OOM_BLOCKED, TASK_NOT_SUPPORTED, ...). This is
# what keeps a tested-and-broken model from being reported as an optimistic
# "*_DERIVED_NEEDS_LIVE_CONFIRMATION" — it gets its true blocker instead.
# >>> BEGIN GENERATED: LIVE_INFERENCE_FAILED
LIVE_INFERENCE_FAILED: dict[str, str] = {
    "florence-2-base": "DEPENDENCY_MISSING",
    "florence-2-large": "DEPENDENCY_MISSING",
    "oneformer-convnext-large": "WEIGHTS_MISSING",
    "oneformer-dinat-large": "DEPENDENCY_MISSING",
}
# <<< END GENERATED: LIVE_INFERENCE_FAILED


def live_inference_verified(model_id: str) -> bool:
    """True iff ``model_id``'s live inference smoke passed this sprint."""
    return model_id in LIVE_INFERENCE_VERIFIED


def live_inference_blocker(model_id: str) -> str | None:
    """Readiness-state blocker for a model live-tested for inference that FAILED."""
    return LIVE_INFERENCE_FAILED.get(model_id)


def live_train_verified(model_id: str) -> bool:
    """True iff ``model_id``'s full live train lifecycle passed this sprint."""
    return model_id in LIVE_TRAIN_VERIFIED


# --------------------------------------------------------------------------- #
# Evidence loaders (used by tools/tests to cross-check the baked conclusions).
# --------------------------------------------------------------------------- #
_DOCS_DIR = Path(__file__).resolve().parents[3] / "docs" / "qa" / "v318_full_model_truth"
INFERENCE_MATRIX_PATH = _DOCS_DIR / "live_inference_matrix.json"
TRAIN_MATRIX_PATH = _DOCS_DIR / "live_train_lifecycle_matrix.json"


def _passed_ids(matrix_path: Path) -> set[str]:
    """Return the set of model ids whose matrix row is a genuine live PASS."""
    if not matrix_path.exists():
        return set()
    data: Any = json.loads(matrix_path.read_text())
    rows = data.get("results", data) if isinstance(data, dict) else data
    out: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("status") == "PASS" and row.get("live_verified") is True:
            out.add(row["model_id"])
    return out


def inference_verified_from_matrix() -> set[str]:
    """The PASS set recorded in the committed live-inference matrix."""
    return _passed_ids(INFERENCE_MATRIX_PATH)


def train_verified_from_matrix() -> set[str]:
    """The PASS set recorded in the committed live-train-lifecycle matrix."""
    return _passed_ids(TRAIN_MATRIX_PATH)


__all__ = [
    "INFERENCE_MATRIX_PATH",
    "LIVE_INFERENCE_FAILED",
    "LIVE_INFERENCE_VERIFIED",
    "LIVE_TRAIN_VERIFIED",
    "TRAIN_MATRIX_PATH",
    "inference_verified_from_matrix",
    "live_inference_blocker",
    "live_inference_verified",
    "live_train_verified",
    "train_verified_from_matrix",
]
