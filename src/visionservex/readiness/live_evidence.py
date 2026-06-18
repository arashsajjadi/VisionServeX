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
        "maxvit-tiny-tf-224",
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
        "rfdetr-base",
        "rfdetr-large",
        "rfdetr-medium",
        "rfdetr-nano",
        "rfdetr-seg-medium",
        "rfdetr-seg-nano",
        "rfdetr-seg-small",
        "rfdetr-small",
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
    "oneformer-convnext-large": "WEIGHTS_MISSING",
    "oneformer-dinat-large": "DEPENDENCY_MISSING",
}
# <<< END GENERATED: LIVE_INFERENCE_FAILED

# v3.20: the reload (from_checkpoint) and export stages were live-proven for these
# models as part of their committed train/finetune lifecycle matrices.
# >>> BEGIN GENERATED: LIVE_RELOAD_VERIFIED
LIVE_RELOAD_VERIFIED: frozenset[str] = frozenset(
    {
        "clip-vit-base-patch32",
        "clip-vit-large-patch14",
        "dinov2-base",
        "dinov2-giant",
        "dinov2-large",
        "dinov2-small",
        "libreyolo-rtdetr-r50",
        "libreyolo-yolov9-s",
        "libreyolo-yolox-s",
        "rfdetr-base",
        "rfdetr-large",
        "rfdetr-medium",
        "rfdetr-nano",
        "rfdetr-seg-medium",
        "rfdetr-seg-nano",
        "rfdetr-seg-small",
        "rfdetr-small",
        "siglip-base-patch16-224",
        "siglip2-base-patch16-224",
        "siglip2-large-patch16-256",
        "siglip2-so400m-patch14-384",
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
# <<< END GENERATED: LIVE_RELOAD_VERIFIED

# >>> BEGIN GENERATED: LIVE_EXPORT_VERIFIED
LIVE_EXPORT_VERIFIED: frozenset[str] = frozenset(
    {
        "libreyolo-rtdetr-r50",
        "libreyolo-yolov9-s",
        "libreyolo-yolox-s",
        "rfdetr-base",
        "rfdetr-large",
        "rfdetr-medium",
        "rfdetr-nano",
        "rfdetr-seg-medium",
        "rfdetr-seg-nano",
        "rfdetr-seg-small",
        "rfdetr-small",
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
# <<< END GENERATED: LIVE_EXPORT_VERIFIED

# v3.20: models whose FINE-TUNE lifecycle (head/adapter/contrastive train ->
# checkpoint -> reload -> embed/similarity-after-reload) passed live this sprint.
# >>> BEGIN GENERATED: LIVE_FINETUNE_VERIFIED
LIVE_FINETUNE_VERIFIED: frozenset[str] = frozenset(
    {
        "clip-vit-base-patch32",
        "clip-vit-large-patch14",
        "dinov2-base",
        "dinov2-giant",
        "dinov2-large",
        "dinov2-small",
        "sam-vit-base",
        "siglip-base-patch16-224",
        "siglip2-base-patch16-224",
        "siglip2-large-patch16-256",
        "siglip2-so400m-patch14-384",
    }
)
# <<< END GENERATED: LIVE_FINETUNE_VERIFIED

# v3.21: models live-verified ONLY through an isolated Docker sidecar — not
# loadable in the default-safe host env, but a real inference smoke passed this
# sprint via the sidecar (e.g. Florence-2 on transformers<5 / py3.11). These earn
# a ``*_READY_LIVE_SIDECAR`` state, which is honestly distinct from host-runnable
# ``*_READY_LIVE`` and supersedes the host DEPENDENCY_MISSING blocker.
# >>> BEGIN GENERATED: LIVE_SIDECAR_VERIFIED
LIVE_SIDECAR_VERIFIED: frozenset[str] = frozenset(
    {
        "florence-2-base",
        "florence-2-large",
        "rtmpose-m",
    }
)
# <<< END GENERATED: LIVE_SIDECAR_VERIFIED


def live_inference_verified(model_id: str) -> bool:
    """True iff ``model_id``'s live inference smoke passed this sprint."""
    return model_id in LIVE_INFERENCE_VERIFIED


def live_inference_blocker(model_id: str) -> str | None:
    """Readiness-state blocker for a model live-tested for inference that FAILED."""
    return LIVE_INFERENCE_FAILED.get(model_id)


def live_train_verified(model_id: str) -> bool:
    """True iff ``model_id``'s full live train lifecycle passed this sprint."""
    return model_id in LIVE_TRAIN_VERIFIED


def live_reload_verified(model_id: str) -> bool:
    """True iff ``model_id``'s checkpoint reload+predict was live-proven this sprint."""
    return model_id in LIVE_RELOAD_VERIFIED


def live_export_verified(model_id: str) -> bool:
    """True iff ``model_id``'s export was live-proven this sprint."""
    return model_id in LIVE_EXPORT_VERIFIED


def live_finetune_verified(model_id: str) -> bool:
    """True iff ``model_id``'s fine-tune lifecycle passed live this sprint."""
    return model_id in LIVE_FINETUNE_VERIFIED


def live_sidecar_verified(model_id: str) -> bool:
    """True iff ``model_id`` ran a live inference smoke via an isolated sidecar."""
    return model_id in LIVE_SIDECAR_VERIFIED


# --------------------------------------------------------------------------- #
# Evidence loaders (used by tools/tests to cross-check the baked conclusions).
# --------------------------------------------------------------------------- #
_DOCS_DIR = Path(__file__).resolve().parents[3] / "docs" / "qa" / "v318_full_model_truth"
_DOCS_DIR_V319 = (
    Path(__file__).resolve().parents[3] / "docs" / "qa" / "v319_operationalize_all_models"
)
INFERENCE_MATRIX_PATH = _DOCS_DIR / "live_inference_matrix.json"
TRAIN_MATRIX_PATH = _DOCS_DIR / "live_train_lifecycle_matrix.json"
# v3.19 operationalization matrices (additive — unioned with the v3.18 evidence).
INFERENCE_MATRIX_PATH_V319 = _DOCS_DIR_V319 / "v319_inference_matrix.json"
RFDETR_TRAIN_MATRIX_PATH = _DOCS_DIR_V319 / "rfdetr_live_train_matrix.json"
# v3.20 final-operationalization matrices.
_DOCS_DIR_V320 = (
    Path(__file__).resolve().parents[3] / "docs" / "qa" / "v320_final_operationalization"
)
TRAIN_FINETUNE_MATRIX_PATH = _DOCS_DIR_V320 / "train_finetune_matrix.json"
INFERENCE_MATRIX_PATH_V320 = _DOCS_DIR_V320 / "v320_inference_matrix.json"
# v3.21 sidecar blocker-elimination matrix (isolated Docker sidecars).
_DOCS_DIR_V321 = (
    Path(__file__).resolve().parents[3] / "docs" / "qa" / "v321_sidecar_blocker_elimination"
)
SIDECAR_MATRIX_PATH_V321 = _DOCS_DIR_V321 / "v321_sidecar_matrix.json"
SEG_FINETUNE_MATRIX_PATH_V321 = _DOCS_DIR_V321 / "v321_segmentation_finetune.json"

_ALL_TRAIN_MATRICES = (TRAIN_MATRIX_PATH, RFDETR_TRAIN_MATRIX_PATH, TRAIN_FINETUNE_MATRIX_PATH)


def _passed_ids(matrix_path: Path) -> set[str]:
    """Return the set of model ids whose matrix row is a genuine live PASS."""
    return {r["model_id"] for r in _rows(matrix_path) if _is_pass(r)}


def _rows(matrix_path: Path) -> list[dict]:
    if not matrix_path.exists():
        return []
    data: Any = json.loads(matrix_path.read_text())
    rows = data.get("results", data) if isinstance(data, dict) else data
    return [r for r in rows if isinstance(r, dict)]


def _is_pass(row: dict) -> bool:
    return row.get("status") == "PASS" and row.get("live_verified") is True


def _stage_ids(matrices, stage: str) -> set[str]:
    """ids whose PASS row has the given lifecycle stage flag True (e.g. reload/export)."""
    out: set[str] = set()
    for mp in matrices:
        for r in _rows(mp):
            if _is_pass(r) and bool(r.get(stage)):
                out.add(r["model_id"])
    return out


def inference_verified_from_matrix() -> set[str]:
    """The PASS set across the committed live-inference matrices (v3.18 + v3.19 + v3.20)."""
    return (
        _passed_ids(INFERENCE_MATRIX_PATH)
        | _passed_ids(INFERENCE_MATRIX_PATH_V319)
        | _passed_ids(INFERENCE_MATRIX_PATH_V320)
    )


def train_verified_from_matrix() -> set[str]:
    """The PASS set across the committed full-train matrices (v3.18 + v3.19 RF-DETR)."""
    return _passed_ids(TRAIN_MATRIX_PATH) | _passed_ids(RFDETR_TRAIN_MATRIX_PATH)


_FINETUNE_METHODS = (
    "fine_tune",
    "adapter_train",
    "head_train",
    "contrastive_train",
    "lora",
    "sam_decoder_finetune",  # v3.21: frozen-encoder SAM mask-decoder fine-tune
)


def finetune_verified_from_matrix() -> set[str]:
    """PASS rows whose method is a fine-tune (v3.20 head/adapter + v3.21 SAM decoder)."""
    out: set[str] = set()
    for mp in (TRAIN_FINETUNE_MATRIX_PATH, SEG_FINETUNE_MATRIX_PATH_V321):
        for r in _rows(mp):
            method = str(r.get("method", ""))
            if _is_pass(r) and any(fm in method for fm in _FINETUNE_METHODS):
                out.add(r["model_id"])
    return out


def sidecar_verified_from_matrix() -> set[str]:
    """PASS set in the committed v3.21 sidecar matrix (live via isolated sidecar)."""
    return _passed_ids(SIDECAR_MATRIX_PATH_V321)


def reload_verified_from_matrix() -> set[str]:
    """ids whose committed lifecycle proved checkpoint reload."""
    return _stage_ids(_ALL_TRAIN_MATRICES, "reload") | _stage_ids(
        _ALL_TRAIN_MATRICES, "reload_verified"
    )


def export_verified_from_matrix() -> set[str]:
    """ids whose committed lifecycle proved export."""
    return _stage_ids(_ALL_TRAIN_MATRICES, "export") | _stage_ids(
        _ALL_TRAIN_MATRICES, "export_verified"
    )


__all__ = [
    "INFERENCE_MATRIX_PATH",
    "LIVE_EXPORT_VERIFIED",
    "LIVE_FINETUNE_VERIFIED",
    "LIVE_INFERENCE_FAILED",
    "LIVE_INFERENCE_VERIFIED",
    "LIVE_RELOAD_VERIFIED",
    "LIVE_SIDECAR_VERIFIED",
    "LIVE_TRAIN_VERIFIED",
    "SIDECAR_MATRIX_PATH_V321",
    "TRAIN_MATRIX_PATH",
    "export_verified_from_matrix",
    "finetune_verified_from_matrix",
    "inference_verified_from_matrix",
    "live_export_verified",
    "live_finetune_verified",
    "live_inference_blocker",
    "live_inference_verified",
    "live_reload_verified",
    "live_sidecar_verified",
    "live_train_verified",
    "reload_verified_from_matrix",
    "sidecar_verified_from_matrix",
    "train_verified_from_matrix",
]
