# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.24.0: Normalize RT-DETRv4 sidecar output → canonical detection rows.

RT-DETRv4's upstream ``tools/inference/torch_inf.py`` emits per-image
``labels`` (int tensor [N]), ``boxes`` (float tensor [N, 4] in xyxy pixel
space), ``scores`` (float tensor [N]). The Transformers-friendly fork
returns the same fields wrapped in a dict; the ONNX/TensorRT export wraps
them in a numpy ndarray of shape [N, 6] = (x1, y1, x2, y2, score, class_id).

Canonical row shape mirrors :mod:`visionservex.sidecars.deimv2_normalize`:

    {"xyxy", "score", "class_id", "category_id", "class_name"}

Invariants:
- Boxes with non-positive width/height are skipped (``n_invalid`` counted).
- Scores < ``score_threshold`` (default 0.0) are dropped.
- ``class_id`` is COCO80 contiguous (0..79). RT-DETRv4 already trains on
  COCO80 contiguous, so no remap is needed.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from visionservex.data.coco_mapping import (
    COCO80_CONTIGUOUS_LABELS,
    COCO_CONTIGUOUS_TO_OFFICIAL,
)

__all__ = [
    "normalize_rtdetrv4_output",
    "RTDETRV4_CANONICAL_FIELDS",
]

RTDETRV4_CANONICAL_FIELDS: tuple[str, ...] = (
    "xyxy",
    "score",
    "class_id",
    "category_id",
    "class_name",
)


def _to_python(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "detach"):
        try:
            obj = obj.detach().cpu().numpy()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return obj


def _finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _coerce_box(box: Iterable[Any]) -> list[float] | None:
    box = list(box)
    if len(box) != 4 or not all(_finite(v) for v in box):
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


def _from_split(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    boxes = _to_python(payload.get("boxes"))
    scores = _to_python(payload.get("scores"))
    labels = _to_python(payload.get("labels"))
    if not isinstance(boxes, list) or not isinstance(scores, list) or not isinstance(labels, list):
        return [], 0
    out: list[dict[str, Any]] = []
    n_invalid = 0
    n = min(len(boxes), len(scores), len(labels))
    for i in range(n):
        bx = _coerce_box(boxes[i] if isinstance(boxes[i], (list, tuple)) else [])
        if bx is None or not _finite(scores[i]) or not _finite(labels[i]):
            n_invalid += 1
            continue
        out.append(_row(bx, float(scores[i]), int(labels[i])))
    return out, n_invalid


def _from_n6(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    n_invalid = 0
    for r in rows:
        if not isinstance(r, (list, tuple)) or len(r) < 6:
            n_invalid += 1
            continue
        bx = _coerce_box(r[:4])
        if bx is None or not _finite(r[4]) or not _finite(r[5]):
            n_invalid += 1
            continue
        out.append(_row(bx, float(r[4]), int(r[5])))
    return out, n_invalid


def normalize_rtdetrv4_output(
    raw: Any,
    *,
    image_id: int | str | None = None,
    score_threshold: float = 0.0,
) -> dict[str, Any]:
    """Normalize RT-DETRv4 raw output into a canonical detection payload.

    Parameters
    ----------
    raw
        The upstream postprocessor output (dict, list-of-dicts, or [N, 6]
        array).
    image_id
        Optional id propagated into the result for traceability.
    score_threshold
        Drop predictions whose score is below this threshold.

    Returns
    -------
    dict
        ``{"status", "code", "image_id", "rows", "n_detections", "n_invalid",
        "score_threshold"}``. ``code="OK"`` for ok;
        ``code="NORMALIZER_OUTPUT_INVALID"`` for unrecognized shapes.
    """
    if raw is None:
        return {
            "status": "ok",
            "code": "OK",
            "image_id": image_id,
            "rows": [],
            "n_detections": 0,
            "n_invalid": 0,
            "score_threshold": float(score_threshold),
        }

    raw = _to_python(raw)

    if isinstance(raw, dict) and {"boxes", "scores", "labels"}.issubset(raw):
        rows, n_invalid = _from_split(raw)
    elif (
        isinstance(raw, list)
        and raw
        and isinstance(raw[0], dict)
        and {"boxes", "scores", "labels"}.issubset(raw[0])
    ):
        rows, n_invalid = _from_split(raw[0])
    elif isinstance(raw, list) and (
        not raw or (isinstance(raw[0], (list, tuple)) and len(raw[0]) >= 6)
    ):
        rows, n_invalid = _from_n6(raw)
    else:
        return {
            "status": "failed",
            "code": "NORMALIZER_OUTPUT_INVALID",
            "image_id": image_id,
            "rows": [],
            "n_detections": 0,
            "n_invalid": 0,
            "raw_type": type(raw).__name__,
            "score_threshold": float(score_threshold),
            "message": (
                "RT-DETRv4 normalizer could not recognise the output shape. "
                "Expected dict with boxes/scores/labels OR list-of-dicts OR "
                "list-of-[x1,y1,x2,y2,score,class_id]."
            ),
        }

    if score_threshold > 0.0:
        rows = [r for r in rows if r["score"] >= score_threshold]

    return {
        "status": "ok",
        "code": "OK",
        "image_id": image_id,
        "rows": rows,
        "n_detections": len(rows),
        "n_invalid": n_invalid,
        "score_threshold": float(score_threshold),
    }
