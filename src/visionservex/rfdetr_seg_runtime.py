# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""RF-DETR instance-segmentation runtime (v3.7 table completion).

Roboflow RF-DETR segmentation variants are Apache-2.0 with auto-downloaded
checkpoints (rfdetr>=1.4). CRITICAL: the *segmentation* XL/2XL checkpoints are
Apache-2.0 and do NOT require the PML-1.0 ``rfdetr_plus`` package (unlike the
detection-only RF-DETR-XL/2XL). All six seg variants are commercial-safe.

Runs on CPU via ``device="cpu"`` so it stays within resource-safety limits.
"""
from __future__ import annotations

from typing import Any

_VARIANTS = {
    "rfdetr-seg-nano": "RFDETRSegNano",
    "rfdetr-seg-small": "RFDETRSegSmall",
    "rfdetr-seg-medium": "RFDETRSegMedium",
    "rfdetr-seg-large": "RFDETRSegLarge",
    "rfdetr-seg-xl": "RFDETRSegXLarge",
    "rfdetr-seg-2xl": "RFDETRSeg2XLarge",
}


def variants() -> list[str]:
    return list(_VARIANTS)


def explain(model_id: str) -> dict[str, Any]:
    if model_id not in _VARIANTS:
        return {"model_id": model_id, "family": "rf-detr", "state": "unknown",
                "next_command": "visionservex segment-instances --help"}
    return {
        "model_id": model_id, "family": "rf-detr", "task": "instance_segmentation",
        "state": "benchmark_passed", "license": "Apache-2.0", "commercial_safe": True,
        "default_safe": True, "install_extra": "visionservex[rfdetr]",
        "backbone": "DINOv2 (Apache-2.0)", "weights": "auto-downloaded (Apache-2.0 seg checkpoint)",
        "output_schema": {"boxes": "xyxy", "masks": "list[HxW bool]", "class_ids": "list[int]",
                          "scores": "list[float]"},
        "limitations": ("XL/2XL are heavier; the SEG XL/2XL are Apache-2.0 and do NOT "
                        "require rfdetr_plus (unlike detection XL/2XL = PML-1.0)."),
        "next_command": f"visionservex segment-instances image.jpg --model {model_id} --out out/",
    }


def segment_instances(model_id: str, image, threshold: float = 0.3,
                      device: str = "cpu", **kw) -> dict[str, Any]:
    """Run RF-DETR instance segmentation. Returns boxes/masks/classes/scores."""
    if model_id not in _VARIANTS:
        raise ValueError(f"unknown RF-DETR-Seg variant {model_id!r}; known: {sorted(_VARIANTS)}")
    import numpy as np
    import rfdetr

    cls = getattr(rfdetr, _VARIANTS[model_id])
    model = cls(device=device)
    det = model.predict(image, threshold=threshold)
    n = int(len(det.xyxy)) if getattr(det, "xyxy", None) is not None else 0
    masks = getattr(det, "mask", None)
    return {
        "model_id": model_id, "engine": "rfdetr", "task": "instance_segmentation",
        "n_instances": n,
        "boxes": det.xyxy.tolist() if n else [],
        "scores": det.confidence.tolist() if getattr(det, "confidence", None) is not None else [],
        "class_ids": det.class_id.tolist() if getattr(det, "class_id", None) is not None else [],
        "has_masks": masks is not None,
        "mask_shape": list(masks.shape) if masks is not None else None,
        "detections": det,  # raw supervision.Detections for downstream use
    }
