# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Detection postprocessing: class-aware NMS / dedup for normalized predict output.

VisionServeX's normalized ``predict()`` must return *final* detections, not raw
proposals. YOLO-family decoders already NMS internally, but DETR-style decoders
(RT-DETR / D-FINE) are set-based and emit up to ``num_queries`` (~300) boxes with
no NMS — an undertrained or low-threshold DETR can therefore flood overlapping
near-duplicates. This module applies a light, class-aware greedy NMS on top of the
engine output (idempotent for already-NMS'd YOLO output) and caps ``max_det``.

Pure numpy — no torch/torchvision dependency, no AGPL/GPL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

# Default IoU for the safety-net NMS. 0.6 is deliberately lenient: it collapses
# clear duplicates without merging genuinely distinct nearby objects.
DEFAULT_NMS_IOU = 0.6
DEFAULT_MAX_DET = 300


def _greedy_nms_single_class(dets: list, iou_thres: float) -> list:
    """Greedy NMS within one class. ``dets`` are objects with ``.box`` + ``.score``."""
    if len(dets) <= 1:
        return list(dets)
    order = sorted(range(len(dets)), key=lambda i: float(dets[i].score), reverse=True)
    boxes = np.array([[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in dets], dtype=np.float64)
    areas = (boxes[:, 2] - boxes[:, 0]).clip(min=0) * (boxes[:, 3] - boxes[:, 1]).clip(min=0)
    suppressed = [False] * len(dets)
    keep: list = []
    for oi in range(len(order)):
        i = order[oi]
        if suppressed[i]:
            continue
        keep.append(dets[i])
        for oj in range(oi + 1, len(order)):
            j = order[oj]
            if suppressed[j]:
                continue
            xx1 = max(boxes[i, 0], boxes[j, 0])
            yy1 = max(boxes[i, 1], boxes[j, 1])
            xx2 = min(boxes[i, 2], boxes[j, 2])
            yy2 = min(boxes[i, 3], boxes[j, 3])
            inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            union = areas[i] + areas[j] - inter
            if union > 0 and (inter / union) > iou_thres:
                suppressed[j] = True
    return keep


def class_aware_nms(
    detections: list,
    *,
    iou_thres: float = DEFAULT_NMS_IOU,
    max_det: int = DEFAULT_MAX_DET,
) -> list:
    """Class-aware greedy NMS + ``max_det`` cap over VisionServeX detections/segments.

    Boxes are suppressed only by others of the SAME ``class_id`` (different classes
    survive). Returns a new list sorted by descending score and capped at
    ``max_det``. Idempotent on already-NMS'd input.
    """
    if not detections:
        return []
    by_class: dict[int, list] = {}
    for d in detections:
        by_class.setdefault(int(getattr(d, "class_id", -1) or -1), []).append(d)
    kept: list = []
    for dets in by_class.values():
        kept.extend(_greedy_nms_single_class(dets, iou_thres))
    kept.sort(key=lambda d: float(d.score), reverse=True)
    return kept[:max_det]


__all__ = ["DEFAULT_MAX_DET", "DEFAULT_NMS_IOU", "class_aware_nms"]
