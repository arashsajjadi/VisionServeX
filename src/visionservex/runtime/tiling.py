# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tiled (sliced) inference for small-object detection (v3.22.0).

Runs the model on overlapping tiles of a large frame, remaps each tile's boxes
(and masks) back to original-frame coordinates, then applies a single cross-tile
class-aware NMS. This recovers small/distant objects that a whole-frame pass
misses, at the cost of more forward passes.

Exposes the raw per-tile detection count and the final post-NMS count so callers
can see exactly what tiling recovered — no hidden behavior.
"""

from __future__ import annotations

from typing import Any

from visionservex.core.results import (
    Box,
    Detection,
    DetectionResult,
    Segment,
    SegmentationResult,
)
from visionservex.runtime.postprocess import class_aware_nms


def _tile_grid(w: int, h: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Compute (x1,y1,x2,y2) tile windows covering the image with overlap."""
    stride = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, max(1, w - tile + 1), stride)) or [0]
    ys = list(range(0, max(1, h - tile + 1), stride)) or [0]
    if xs[-1] + tile < w:
        xs.append(w - tile)
    if ys[-1] + tile < h:
        ys.append(h - tile)
    windows = []
    for y in ys:
        for x in xs:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x1 + tile)
            y2 = min(h, y1 + tile)
            windows.append((x1, y1, x2, y2))
    return windows


def tiled_predict(
    model: Any,
    image: Any,
    *,
    tile: int = 640,
    overlap: float = 0.2,
    nms_iou: float = 0.6,
    max_det: int = 1000,
    use_batch: bool = True,
    **predict_kwargs: Any,
) -> DetectionResult | SegmentationResult:
    """Run sliced inference and merge with cross-tile class-aware NMS.

    Works for detection and instance segmentation. Returns the same result type
    as the model, with ``metadata`` carrying ``tile_count``, ``raw_count`` and
    ``final_count``.
    """
    w, h = image.size
    windows = _tile_grid(w, h, tile, overlap)
    crops = [image.crop(win) for win in windows]

    # Run tiles (true batch if the model supports it, else loop).
    if use_batch and hasattr(model, "batch_predict"):
        results = model.batch_predict(crops, **predict_kwargs)
    else:
        results = [model.predict(c, **predict_kwargs) for c in crops]

    is_seg = any(isinstance(r, SegmentationResult) for r in results)
    merged_dets: list[Detection] = []
    merged_segs: list[Segment] = []
    raw_count = 0

    for (ox, oy, _x2, _y2), res in zip(windows, results, strict=True):
        if isinstance(res, SegmentationResult):
            for s in res.segments:
                raw_count += 1
                b = s.box
                merged_segs.append(
                    Segment(
                        box=Box(b.x1 + ox, b.y1 + oy, b.x2 + ox, b.y2 + oy),
                        score=s.score,
                        label=s.label,
                        mask=_remap_mask(s.mask, ox, oy, w, h),
                        class_id=s.class_id,
                    )
                )
        else:
            for d in getattr(res, "detections", []):
                raw_count += 1
                b = d.box
                merged_dets.append(
                    Detection(
                        box=Box(b.x1 + ox, b.y1 + oy, b.x2 + ox, b.y2 + oy),
                        score=d.score,
                        label=d.label,
                        class_id=d.class_id,
                    )
                )

    if is_seg:
        kept = class_aware_nms(merged_segs, iou_thres=nms_iou, max_det=max_det)
        out: SegmentationResult | DetectionResult = SegmentationResult(
            kind="segmentation",
            model_id=getattr(model, "entry", None).id if getattr(model, "entry", None) else "",
            task="segment",
            image_size=(w, h),
            segments=kept,
        )
    else:
        kept = class_aware_nms(merged_dets, iou_thres=nms_iou, max_det=max_det)
        out = DetectionResult(
            kind="detection",
            model_id=getattr(model, "entry", None).id if getattr(model, "entry", None) else "",
            task="detect",
            image_size=(w, h),
            detections=kept,
        )
    out.metadata.update(
        {
            "mode": "tiled",
            "tile": tile,
            "overlap": overlap,
            "tile_count": len(windows),
            "raw_count": raw_count,
            "final_count": len(kept),
        }
    )
    out._image = image
    return out


def _remap_mask(mask: Any, ox: int, oy: int, w: int, h: int) -> Any:
    """Paste a tile mask into a full-frame canvas at (ox, oy)."""
    import numpy as np

    if getattr(mask, "size", 0) <= 1:
        return mask
    canvas = np.zeros((h, w), dtype=np.uint8)
    mh, mw = mask.shape[:2]
    canvas[oy : oy + mh, ox : ox + mw] = mask[: h - oy, : w - ox]
    return canvas


def frame_diagnostics(
    result: Any,
    *,
    raw_candidates: int | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Per-frame detection accounting: raw → kept → final, plus an empty reason.

    ``raw_candidates`` is the model's pre-threshold query count when known
    (e.g. 300 for D-FINE/DETR). Returns a JSON-safe dict.
    """
    dets = getattr(result, "detections", None)
    if dets is None:
        dets = getattr(result, "segments", []) or []
    final = len(dets)
    scores = sorted((float(getattr(d, "score", 0.0)) for d in dets), reverse=True)
    labels: dict[str, int] = {}
    sizes: list[float] = []
    for d in dets:
        labels[getattr(d, "label", "?")] = labels.get(getattr(d, "label", "?"), 0) + 1
        b = getattr(d, "box", None)
        if b is not None:
            sizes.append(float(b.area))

    empty_reason = None
    if final == 0:
        if raw_candidates and threshold is not None:
            empty_reason = f"all {raw_candidates} raw candidates scored below threshold {threshold}"
        elif threshold is not None:
            empty_reason = f"no detections above threshold {threshold}"
        else:
            empty_reason = "model returned no detections"

    return {
        "raw_candidates": raw_candidates,
        "threshold": threshold,
        "final_count": final,
        "class_distribution": labels,
        "score_min": round(scores[-1], 4) if scores else None,
        "score_max": round(scores[0], 4) if scores else None,
        "size_min_px": round(min(sizes), 1) if sizes else None,
        "size_max_px": round(max(sizes), 1) if sizes else None,
        "empty_reason": empty_reason,
    }


__all__ = ["frame_diagnostics", "tiled_predict"]
