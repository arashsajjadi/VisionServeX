# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.16.0: class-aware NMS / duplicate-box suppression for normalized predict."""

from __future__ import annotations

from visionservex.core.results import Box, Detection
from visionservex.runtime.postprocess import class_aware_nms


def _det(x1, y1, x2, y2, score, cid, label="o"):
    return Detection(box=Box(x1=x1, y1=y1, x2=x2, y2=y2), score=score, label=label, class_id=cid)


def test_overlapping_same_class_collapse_to_one():
    dets = [
        _det(0, 0, 10, 10, 0.9, 0),
        _det(1, 1, 11, 11, 0.8, 0),  # ~81% IoU with the first
        _det(0, 0, 10, 10, 0.7, 0),  # identical to the first
    ]
    out = class_aware_nms(dets, iou_thres=0.5)
    assert len(out) == 1
    assert out[0].score == 0.9  # highest-scoring survivor kept


def test_different_class_boxes_survive():
    dets = [_det(0, 0, 10, 10, 0.9, 0), _det(0, 0, 10, 10, 0.8, 1)]  # same box, different class
    out = class_aware_nms(dets, iou_thres=0.5)
    assert len(out) == 2  # class-aware: different classes are not suppressed


def test_non_overlapping_all_survive():
    dets = [_det(i * 100, 0, i * 100 + 10, 10, 0.5, 0) for i in range(5)]
    out = class_aware_nms(dets, iou_thres=0.5)
    assert len(out) == 5


def test_max_det_cap_and_sorted():
    dets = [_det(i * 100, 0, i * 100 + 10, 10, i / 10.0, 0) for i in range(10)]
    out = class_aware_nms(dets, iou_thres=0.5, max_det=3)
    assert len(out) == 3
    assert [d.score for d in out] == [0.9, 0.8, 0.7]  # top-3 by score


def test_empty_input():
    assert class_aware_nms([]) == []


def test_engine_predict_exposes_nms_and_raw_params():
    """The libreyolo engine predict() must accept nms_iou / max_det / return_raw."""
    import inspect

    from visionservex.engines.libreyolo import LibreYOLOEngine

    params = inspect.signature(LibreYOLOEngine.predict).parameters
    assert "nms_iou" in params
    assert "max_det" in params
    assert "return_raw" in params
