# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Mask serialization: COCO RLE + polygon contours + quality checks (v3.22.0).

This is the fix for "segmentation returns boxes only": segmentation masks ARE
produced by the engines (``Segment.mask``) but were dropped at JSON time. These
helpers turn a HxW uint8 mask into a transmittable, machine-readable form:

* :func:`mask_to_rle`      — compact COCO RLE (pycocotools if available, else a
  pure-python uncompressed COCO RLE that pycocotools/most tools can still read).
* :func:`mask_to_polygons` — contour polygons (cv2), optionally simplified and
  point-capped, so a UI can render vector outlines cheaply.
* :func:`mask_quality`     — area / box-alignment / tiny-or-giant warnings.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_uint8(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8)
    return m


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask to COCO RLE ``{size:[H,W], counts:str}``.

    Uses pycocotools (compressed RLE) when present; otherwise emits a portable
    uncompressed COCO RLE (column-major run lengths) that is still valid COCO.
    """
    m = _as_uint8(mask)
    if m.ndim != 2 or m.size <= 1:
        return {"size": list(m.shape), "counts": "", "format": "empty"}
    try:
        from pycocotools import mask as mask_utils  # type: ignore

        rle = mask_utils.encode(np.asfortranarray(m))
        counts = rle["counts"]
        return {
            "size": [int(m.shape[0]), int(m.shape[1])],
            "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts,
            "format": "coco_rle",
        }
    except Exception:
        # Pure-python uncompressed COCO RLE (column-major).
        flat = np.asfortranarray(m).ravel(order="F")
        counts: list[int] = []
        # COCO RLE starts with a run of 0s.
        prev = 0
        run = 0
        for v in flat:
            if v == prev:
                run += 1
            else:
                counts.append(run)
                run = 1
                prev = v
        counts.append(run)
        return {
            "size": [int(m.shape[0]), int(m.shape[1])],
            "counts": counts,
            "format": "uncompressed_rle",
        }


def mask_to_polygons(
    mask: np.ndarray,
    *,
    max_points: int = 0,
    tolerance: float = 1.0,
) -> list[list[float]]:
    """Extract polygon contours ``[[x1,y1,x2,y2,...], ...]`` from a binary mask.

    ``tolerance`` controls Douglas-Peucker simplification (cv2.approxPolyDP).
    ``max_points`` (>0) further down-samples each polygon's vertex count.
    """
    m = _as_uint8(mask)
    if m.ndim != 2 or m.size <= 1:
        return []
    try:
        import cv2  # type: ignore
    except Exception:
        return []
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[list[float]] = []
    for c in contours:
        if len(c) < 3:
            continue
        if tolerance and tolerance > 0:
            eps = tolerance
            c = cv2.approxPolyDP(c, eps, True)
        pts = c.reshape(-1, 2).astype(float)
        if max_points and max_points > 0 and len(pts) > max_points:
            idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
            pts = pts[idx]
        polys.append([float(v) for xy in pts for v in xy])
    return polys


def mask_quality(
    mask: np.ndarray, box: Any = None, *, image_area: float | None = None
) -> dict[str, Any]:
    """Compute mask area + box-alignment + tiny/giant/invalid warnings."""
    m = _as_uint8(mask)
    warnings: list[str] = []
    if m.ndim != 2 or m.size <= 1:
        return {"valid": False, "area_px": 0, "warnings": ["invalid_or_empty_mask"]}
    area = int(np.count_nonzero(m))
    h, w = m.shape
    frame_area = float(image_area) if image_area else float(h * w)
    frac = area / frame_area if frame_area else 0.0
    if area == 0:
        warnings.append("empty_mask")
    elif frac < 0.0005:
        warnings.append("tiny_mask")
    elif frac > 0.95:
        warnings.append("giant_mask")

    box_iou = None
    if box is not None and area > 0:
        ys, xs = np.where(m)
        mx1, my1, mx2, my2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        bx1, by1, bx2, by2 = box.x1, box.y1, box.x2, box.y2
        ix1, iy1 = max(mx1, bx1), max(my1, by1)
        ix2, iy2 = min(mx2, bx2), min(my2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        a_mask = (mx2 - mx1) * (my2 - my1)
        a_box = (bx2 - bx1) * (by2 - by1)
        union = a_mask + a_box - inter
        box_iou = inter / union if union > 0 else 0.0
        if box_iou < 0.3:
            warnings.append("mask_box_misaligned")

    return {
        "valid": area > 0,
        "area_px": area,
        "area_frac": round(frac, 5),
        "mask_box_iou": round(box_iou, 3) if box_iou is not None else None,
        "warnings": warnings,
    }


__all__ = ["mask_quality", "mask_to_polygons", "mask_to_rle"]
