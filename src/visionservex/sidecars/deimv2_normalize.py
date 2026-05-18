# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.24.0: Normalize DEIMv2 sidecar output → canonical detection rows.

DEIMv2's upstream postprocessor (``engine.deim.postprocessor.PostProcessor``)
emits a list-of-dicts per image with ``labels`` (Long tensor of contiguous
COCO80 indices), ``boxes`` (Float tensor [N, 4] in xyxy pixel space) and
``scores`` (Float tensor [N]). Some forks emit a single concatenated tensor
[N, 6] = (x1, y1, x2, y2, score, class_id). We accept either shape plus the
"raw numpy / list" path that lab notebooks tend to produce.

Canonical row:

    {
        "xyxy": [x1, y1, x2, y2],     # absolute pixel coordinates
        "score": float in [0, 1],
        "class_id": int in [0, 79],   # COCO80 contiguous
        "category_id": int in [1, 90],# official COCO category id (with gaps)
        "class_name": str,            # from COCO80_CONTIGUOUS_LABELS
    }

Boxes whose width or height is non-positive, or whose score is NaN, are
skipped and counted under ``n_invalid``. The function NEVER raises on
malformed input — it returns a structured payload with
``code="NORMALIZER_OUTPUT_INVALID"`` so the sidecar's failure is visible
upstream.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable
from typing import Any

from visionservex.data.coco_mapping import (
    COCO80_CONTIGUOUS_LABELS,
    COCO_CONTIGUOUS_TO_OFFICIAL,
)

__all__ = [
    "DEIMV2_CANONICAL_FIELDS",
    "normalize_deimv2_output",
]

DEIMV2_CANONICAL_FIELDS: tuple[str, ...] = (
    "xyxy",
    "score",
    "class_id",
    "category_id",
    "class_name",
)


def _to_python(obj: Any) -> Any:
    """Convert torch/numpy scalars and iterables to plain Python."""
    if obj is None:
        return None
    if hasattr(obj, "detach"):
        with contextlib.suppress(Exception):
            obj = obj.detach().cpu().numpy()
    if hasattr(obj, "tolist"):
        with contextlib.suppress(Exception):
            return obj.tolist()
    return obj


def _is_finite_number(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _coerce_box(box: Iterable[Any]) -> list[float] | None:
    box = list(box)
    if len(box) != 4:
        return None
    if not all(_is_finite_number(v) for v in box):
        return None
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _row(box: list[float], score: float, class_id: int) -> dict[str, Any]:
    cid = int(class_id) if 0 <= int(class_id) < len(COCO80_CONTIGUOUS_LABELS) else -1
    return {
        "xyxy": box,
        "score": float(score),
        "class_id": cid,
        "category_id": COCO_CONTIGUOUS_TO_OFFICIAL.get(cid, -1) if cid >= 0 else -1,
        "class_name": COCO80_CONTIGUOUS_LABELS[cid] if cid >= 0 else "unknown",
    }


def _from_concat_n6(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    n_invalid = 0
    for r in rows:
        if not isinstance(r, (list, tuple)) or len(r) < 6:
            n_invalid += 1
            continue
        box = _coerce_box(r[:4])
        if box is None or not _is_finite_number(r[4]) or not _is_finite_number(r[5]):
            n_invalid += 1
            continue
        out.append(_row(box, float(r[4]), int(r[5])))
    return out, n_invalid


def _from_split_dict(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    boxes = _to_python(payload.get("boxes"))
    scores = _to_python(payload.get("scores"))
    labels = _to_python(payload.get("labels"))
    if not isinstance(boxes, list) or not isinstance(scores, list) or not isinstance(labels, list):
        return [], 0
    n = min(len(boxes), len(scores), len(labels))
    out: list[dict[str, Any]] = []
    n_invalid = 0
    for i in range(n):
        box = _coerce_box(boxes[i] if isinstance(boxes[i], (list, tuple)) else [])
        if box is None or not _is_finite_number(scores[i]) or not _is_finite_number(labels[i]):
            n_invalid += 1
            continue
        out.append(_row(box, float(scores[i]), int(labels[i])))
    return out, n_invalid


def normalize_deimv2_output(raw: Any, *, image_id: int | str | None = None) -> dict[str, Any]:
    """Normalize DEIMv2 raw output into a canonical detection payload.

    Returns
    -------
    dict
        ``{"status": "ok"|"failed", "code": str, "image_id": ..., "rows":
        [...], "n_detections": int, "n_invalid": int}``.
        ``code="OK"`` for ok; ``code="NORMALIZER_OUTPUT_INVALID"`` when
        the shape is unrecognized.
    """
    if raw is None:
        return {
            "status": "ok",
            "code": "OK",
            "image_id": image_id,
            "rows": [],
            "n_detections": 0,
            "n_invalid": 0,
        }

    raw = _to_python(raw)

    # Shape A: a dict with boxes/scores/labels
    if isinstance(raw, dict) and {"boxes", "scores", "labels"}.issubset(raw):
        rows, n_invalid = _from_split_dict(raw)
        return {
            "status": "ok",
            "code": "OK",
            "image_id": image_id,
            "rows": rows,
            "n_detections": len(rows),
            "n_invalid": n_invalid,
        }

    # Shape B: list of per-image dicts (typical batched output) — pick the
    # first one if no image_id is given.
    if (
        isinstance(raw, list)
        and raw
        and isinstance(raw[0], dict)
        and {"boxes", "scores", "labels"}.issubset(raw[0])
    ):
        rows, n_invalid = _from_split_dict(raw[0])
        return {
            "status": "ok",
            "code": "OK",
            "image_id": image_id,
            "rows": rows,
            "n_detections": len(rows),
            "n_invalid": n_invalid,
        }

    # Shape C: list of per-detection [x1,y1,x2,y2,score,class_id]
    if isinstance(raw, list) and (
        not raw or (isinstance(raw[0], (list, tuple)) and len(raw[0]) >= 6)
    ):
        rows, n_invalid = _from_concat_n6(raw)
        return {
            "status": "ok",
            "code": "OK",
            "image_id": image_id,
            "rows": rows,
            "n_detections": len(rows),
            "n_invalid": n_invalid,
        }

    return {
        "status": "failed",
        "code": "NORMALIZER_OUTPUT_INVALID",
        "image_id": image_id,
        "rows": [],
        "n_detections": 0,
        "n_invalid": 0,
        "raw_type": type(raw).__name__,
        "message": (
            "DEIMv2 normalizer could not recognise the output shape. "
            "Expected dict with boxes/scores/labels OR list-of-dicts OR "
            "list-of-[x1,y1,x2,y2,score,class_id]."
        ),
    }
