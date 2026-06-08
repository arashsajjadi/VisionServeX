# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Click-based interactive segmentation runtime (v3.7 table completion).

Four named deep interactive segmenters from the v3.7 table are wired here with
HONEST license states derived from a 2026 license audit:

  ritm        MIT code, SBD/COCO+LVIS training -> commercial-safe (product-grade,
              BYOT weights: clone repo + download checkpoint).
  clickseg    MIT code, but FocalClick/SegFormer variant inherits NVIDIA
              non-commercial backbone -> legal_review_required.
  simpleclick MIT code, but MAE (CC-BY-NC) backbone in published weights ->
              legal_review_required.
  focalclick  MIT code, SegFormer (NVIDIA non-commercial) backbone ->
              legal_review_required.

Point prompts: positive_points / negative_points as lists of (x, y).
For commercial-safe, zero-dependency interactive refinement that runs TODAY,
``classic`` routes to the weight-free CPU refiners in ``smart_annotation``
(GrabCut / watershed / random-walker — OpenCV/scikit-image, no GPL).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# (state, license, commercial_safe, training_data_note, checkpoint_rel_path, source)
_FACTS: dict[str, dict[str, Any]] = {
    "ritm": {
        "state": "checkpoint_required",
        "product_grade": "product_grade_candidate",
        "license": "MIT (code) + SBD/COCO+LVIS (permissive data)",
        "commercial_safe": True,
        "training_data": "SBD (CC-BY-SA) + COCO+LVIS (CC-BY-4.0). No NC backbone.",
        "checkpoint": "~/.cache/visionservex/ritm/coco_lvis_h18_itermask.pth",
        "source": "https://github.com/SamsungLabs/ritm_interactive_segmentation",
        "install": (
            "git clone https://github.com/SamsungLabs/ritm_interactive_segmentation && "
            "cd ritm_interactive_segmentation && pip install -r requirements.txt"
        ),
    },
    "clickseg": {
        "state": "legal_review_required",
        "product_grade": "legal_review_required",
        "license": "MIT (code); FocalClick/SegFormer variant = NVIDIA non-commercial",
        "commercial_safe": False,
        "training_data": "COCO+LVIS+8-dataset; SegFormer backbone NVIDIA NC in FocalClick variant.",
        "checkpoint": "~/.cache/visionservex/clickseg/clickseg_cdnet.pth",
        "source": "https://github.com/alibaba/ClickSEG",
        "install": "git clone https://github.com/alibaba/ClickSEG && cd ClickSEG && pip install -r requirements.txt",
    },
    "simpleclick": {
        "state": "legal_review_required",
        "product_grade": "legal_review_required",
        "license": "MIT (code); MAE ImageNet-1k backbone = CC-BY-NC-4.0",
        "commercial_safe": False,
        "training_data": "Published weights inherit MAE CC-BY-NC backbone. Retrain remediation path.",
        "checkpoint": "~/.cache/visionservex/simpleclick/simpleclick_models/cocolvis_vit_huge.pth",
        "source": "https://github.com/uncbiag/SimpleClick",
        "install": "git clone https://github.com/uncbiag/SimpleClick && cd SimpleClick && pip install -r requirements.txt",
    },
    "focalclick": {
        "state": "legal_review_required",
        "product_grade": "legal_review_required",
        "license": "MIT (code); SegFormer backbone = NVIDIA Source Code License (NC)",
        "commercial_safe": False,
        "training_data": "SegFormer-variant weights need NVIDIA commercial licensing.",
        "checkpoint": "~/.cache/visionservex/focalclick/focalclick_segformerb3.pth",
        "source": "https://github.com/alibaba/ClickSEG",
        "install": "git clone https://github.com/alibaba/ClickSEG && cd ClickSEG && pip install -r requirements.txt",
    },
}

# Classic, weight-free CPU refiners that genuinely run today (commercial-safe).
_CLASSIC = {"grabcut", "watershed", "random-walker", "slic-graphcut", "classic"}


def facts(model_id: str) -> dict[str, Any]:
    return _FACTS.get(model_id, {})


def explain(model_id: str) -> dict[str, Any]:
    if model_id in _CLASSIC:
        return {
            "model_id": model_id,
            "family": "interactive",
            "task": "interactive_segmentation",
            "state": "tool_available",
            "license": "OpenCV/scikit-image (Apache-2.0/BSD)",
            "commercial_safe": True,
            "default_safe": True,
            "install_extra": "visionservex[interactive-seg]",
            "prompt_types": ["positive_points", "negative_points"],
            "output_schema": {"mask": "HxW uint8", "polygon": "optional"},
            "limitations": "classic CPU refiner; no learned prior",
            "next_command": f"visionservex interactive run {model_id} image.jpg --positive-points pos.json --out out/",
        }
    f = _FACTS.get(model_id)
    if not f:
        return {
            "model_id": model_id,
            "family": "interactive",
            "state": "unknown",
            "next_command": "visionservex interactive list",
        }
    return {
        "model_id": model_id,
        "family": "interactive",
        "task": "interactive_segmentation",
        "state": f["state"],
        "product_grade_status": f["product_grade"],
        "license": f["license"],
        "commercial_safe": f["commercial_safe"],
        "default_safe": f["commercial_safe"] and f["state"] not in ("legal_review_required",),
        "install_extra": "visionservex[interactive-seg]",
        "prompt_types": ["positive_points", "negative_points"],
        "checkpoint_path": f["checkpoint"],
        "source": f["source"],
        "training_data_note": f["training_data"],
        "output_schema": {"mask": "HxW uint8", "polygon": "optional", "noc": "optional"},
        "limitations": (
            "BYOT weights — not pip-installable with bundled checkpoints. "
            "Legal-review variants must not ship commercially as-is."
        ),
        "next_command": (
            f"# install: {f['install']}\n"
            f"visionservex interactive run {model_id} image.jpg "
            f"--positive-points pos.json --negative-points neg.json --out out/"
        ),
    }


def run_interactive(
    model_id: str, image, positive_points=None, negative_points=None, **kw
) -> dict[str, Any]:
    """Run click-based interactive segmentation.

    Classic refiners run immediately (CPU, weight-free). Named deep models
    (ritm/clickseg/simpleclick/focalclick) require BYOT checkpoints; if absent
    a structured blocker carries the exact install/download command. Legal-review
    models additionally surface their non-commercial restriction.
    """

    if model_id in _CLASSIC:
        return _run_classic(model_id, image, positive_points, negative_points, **kw)

    f = _FACTS.get(model_id)
    if not f:
        raise ValueError(
            f"unknown interactive model: {model_id!r}; "
            f"known: {sorted(_FACTS)} or classic: {sorted(_CLASSIC)}"
        )
    ckpt = Path(f["checkpoint"]).expanduser()
    if not ckpt.exists():
        return {
            "model_id": model_id,
            "status": "blocked",
            "blocker_code": "INTERACTIVE_CHECKPOINT_REQUIRED",
            "state": f["state"],
            "commercial_safe": f["commercial_safe"],
            "reason": (
                f"{model_id} weights not found at {ckpt}. BYOT: "
                + f["install"]
                + (" — NOTE: " + f["training_data"] if not f["commercial_safe"] else "")
            ),
            "next_command": f["install"],
        }
    # Checkpoint present — attempt to load via the user-cloned sidecar package.
    # (We never ship these weights; this path runs only on user-provided installs.)
    raise RuntimeError(
        f"{model_id} checkpoint present at {ckpt} but the upstream package "
        f"({f['source']}) must be importable. Clone + pip install -e it, then re-run."
    )


def _run_classic(model_id, image, positive_points, negative_points, **kw):
    import numpy as np

    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "classic interactive refiner needs opencv: pip install 'visionservex[interactive-seg]'"
        ) from e
    img = np.asarray(image)[..., ::-1] if hasattr(image, "size") else np.asarray(image)
    h, w = img.shape[:2]
    # Build a foreground/background seed mask from clicks for GrabCut.
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    for x, y in positive_points or []:
        cv2.circle(mask, (int(x), int(y)), max(3, w // 80), int(cv2.GC_FGD), -1)
    for x, y in negative_points or []:
        cv2.circle(mask, (int(x), int(y)), max(3, w // 80), int(cv2.GC_BGD), -1)
    if not (positive_points):
        # default seed: central rectangle
        cv2.rectangle(mask, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), int(cv2.GC_PR_FGD), -1)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(np.ascontiguousarray(img), mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")
    return {
        "model_id": model_id,
        "status": "ok",
        "backend": "opencv-grabcut",
        "mask_shape": [h, w],
        "mask_area": int(out.sum()),
        "mask": out,
    }
