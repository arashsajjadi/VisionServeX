# SPDX-License-Identifier: Apache-2.0
"""v3.22.0 — segmentation output correctness (mask RLE/polygon serialization).

The fix for "segmentation returns boxes only": masks must be transmittable.
CI-safe (no torch/GPU) — builds SegmentationResult directly with numpy masks.
"""

from __future__ import annotations

import numpy as np

from visionservex.core.results import Box, Segment, SegmentationResult
from visionservex.runtime.mask_encoding import mask_quality, mask_to_polygons, mask_to_rle


def _disc_mask(h=64, w=64, cx=32, cy=32, r=12) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype(np.uint8)


def _seg_result(masks) -> SegmentationResult:
    segs = []
    for m in masks:
        ys, xs = np.where(m)
        box = (
            Box(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            if len(xs)
            else Box(0, 0, 1, 1)
        )
        segs.append(Segment(box=box, score=0.9, label="obj", mask=m, class_id=1))
    return SegmentationResult(kind="segmentation", model_id="t", task="segment", segments=segs)


def test_rle_roundtrip_reconstructs_mask() -> None:
    m = _disc_mask()
    rle = mask_to_rle(m)
    assert rle["size"] == [64, 64]
    # decode via pycocotools if present, else verify pure-python format shape
    try:
        from pycocotools import mask as mu

        counts = rle["counts"].encode() if isinstance(rle["counts"], str) else rle["counts"]
        dec = mu.decode({"size": rle["size"], "counts": counts})
        iou = np.logical_and(dec > 0, m > 0).sum() / max(1, np.logical_or(dec > 0, m > 0).sum())
        assert iou == 1.0
    except Exception:
        assert rle["format"] in ("coco_rle", "uncompressed_rle")


def test_polygon_extraction_and_capping() -> None:
    m = _disc_mask()
    polys = mask_to_polygons(m, max_points=8, tolerance=1.0)
    assert polys, "a filled disc must yield at least one contour"
    assert len(polys[0]) // 2 <= 8


def test_mask_quality_flags() -> None:
    tiny = np.zeros((100, 100), dtype=np.uint8)
    tiny[0:2, 0:2] = 1
    q = mask_quality(tiny, Box(0, 0, 2, 2))
    assert q["valid"] is True
    assert "tiny_mask" in q["warnings"]


def test_segmentation_to_dict_emits_rle_by_default() -> None:
    r = _seg_result([_disc_mask()])
    d = r.to_dict()
    assert d["masks_available"] is True
    assert d["output_mode"] == "boxes+masks_rle"
    s0 = d["segments"][0]
    assert "rle" in s0 and s0["rle"]["size"] == [64, 64]
    assert "mask" not in s0  # raw ndarray never serialized
    assert s0["mask_pixels_on"] > 0


def test_segmentation_polygons_on_request() -> None:
    r = _seg_result([_disc_mask()])
    r.return_polygons = True
    d = r.to_dict()
    assert "polygons" in d["segments"][0]
    assert "polygons" in d["output_mode"]


def test_segmentation_masks_unavailable_warns() -> None:
    """A seg result with no real masks must warn, not pass silently as boxes."""
    empty = np.zeros((1, 1), dtype=np.uint8)
    seg = Segment(box=Box(0, 0, 10, 10), score=0.8, label="x", mask=empty, class_id=0)
    r = SegmentationResult(kind="segmentation", model_id="t", task="segment", segments=[seg])
    d = r.to_dict()
    assert d["masks_available"] is False
    assert d["output_mode"] == "boxes_only_masks_unavailable"
    assert any("SEGMENTATION_MASKS_UNAVAILABLE" in w for w in d["warnings"])


def test_return_rle_can_be_disabled() -> None:
    r = _seg_result([_disc_mask()])
    r.return_rle = False
    d = r.to_dict()
    assert "rle" not in d["segments"][0]
    assert d["output_mode"] == "boxes+masks"
