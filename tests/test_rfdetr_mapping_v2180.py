# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Phase 1 (v2.18.0): RF-DETR class-mapping fix.

The v17 probe reported AP50 class-aware=0.005 vs class-agnostic=0.854 — a
0.85 gap that proved RF-DETR was returning official COCO ids (1..90) and
the engine was indexing a contiguous label table. v2.18.0 fixes the engine
to remap official → contiguous before assembling Detection objects.

These tests pin the mapping table and the engine helper.
"""

from __future__ import annotations

import numpy as np

from visionservex.data.coco_mapping import (
    COCO80_CONTIGUOUS_LABELS,
    COCO_CONTIGUOUS_TO_OFFICIAL,
    COCO_OFFICIAL_TO_CONTIGUOUS,
    is_official_id_set,
    remap_official_to_contiguous,
)
from visionservex.engines.rfdetr import _sv_to_detections

# ---------------------------------------------------------------------------
# coco_mapping table sanity
# ---------------------------------------------------------------------------


def test_table_has_exactly_80_classes() -> None:
    assert len(COCO80_CONTIGUOUS_LABELS) == 80
    assert len(COCO_OFFICIAL_TO_CONTIGUOUS) == 80
    assert len(COCO_CONTIGUOUS_TO_OFFICIAL) == 80


def test_official_id_1_is_person() -> None:
    cid, label, src = remap_official_to_contiguous(1)
    assert cid == 0
    assert label == "person"
    assert src == "coco_official_to_contiguous"


def test_official_id_3_is_car() -> None:
    cid, label, src = remap_official_to_contiguous(3)
    assert cid == 2
    assert label == "car"
    assert src == "coco_official_to_contiguous"


def test_official_id_88_is_teddy_bear() -> None:
    cid, label, src = remap_official_to_contiguous(88)
    assert cid == 77
    assert label == "teddy bear"
    assert src == "coco_official_to_contiguous"


def test_official_id_90_is_toothbrush() -> None:
    cid, label, _ = remap_official_to_contiguous(90)
    assert cid == 79
    assert label == "toothbrush"


def test_official_id_47_is_cup_not_carrot() -> None:
    """The v17 bug: official id 47 (cup) was being labelled 'carrot'."""
    cid, label, src = remap_official_to_contiguous(47)
    assert label == "cup", label
    assert cid == 41
    assert src == "coco_official_to_contiguous"


def test_official_id_53_is_apple() -> None:
    cid, label, _ = remap_official_to_contiguous(53)
    assert label == "apple"
    assert cid == 47


def test_contiguous_id_0_is_already_contiguous() -> None:
    cid, label, src = remap_official_to_contiguous(0)
    # 0 is NOT a valid official id (official starts at 1), so it falls
    # into the "already_contiguous" branch.
    assert cid == 0
    assert label == "person"
    assert src == "already_contiguous"


def test_unknown_id_returns_minus_one() -> None:
    cid, label, src = remap_official_to_contiguous(999)
    assert cid == -1
    assert src == "unknown"
    assert "unknown" in label.lower()


def test_is_official_id_set_detects_id_above_79() -> None:
    assert is_official_id_set([88]) is True
    assert is_official_id_set([40, 46, 88]) is True


def test_is_official_id_set_returns_false_for_contiguous() -> None:
    assert is_official_id_set([0, 5, 79]) is False
    assert is_official_id_set([]) is False


# ---------------------------------------------------------------------------
# Engine integration: _sv_to_detections with a fake supervision.Detections
# ---------------------------------------------------------------------------


class _FakeSvDetections:
    """Minimal stand-in for supervision.Detections that the engine helper expects."""

    def __init__(self, xyxy, confidence, class_id, data=None):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id
        self.data = data or {}

    def __len__(self):
        return len(self.xyxy)


def test_engine_remaps_official_ids_to_contiguous() -> None:
    """The v17 RF-DETR bug: engine sees official id 47, must produce 'cup' / contiguous 41."""
    dets = _FakeSvDetections(
        xyxy=np.array([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0]]),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([47, 88]),  # official: cup + teddy bear
        data={
            # rfdetr's bad name array — must be IGNORED when official ids detected.
            "class_name": ["carrot", "toilet"]
        },
    )
    out = _sv_to_detections(dets, list(COCO80_CONTIGUOUS_LABELS))
    assert len(out) == 2
    # Contiguous ids, correct labels — NOT the rfdetr names_arr garbage.
    assert out[0].class_id == 41
    assert out[0].label == "cup"
    assert out[1].class_id == 77
    assert out[1].label == "teddy bear"


def test_engine_passes_contiguous_ids_through() -> None:
    """If all ids are 0..79 with no >79 value, they're contiguous and pass through."""
    dets = _FakeSvDetections(
        xyxy=np.array([[10.0, 10.0, 50.0, 50.0]]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),  # person, contiguous
        data={},
    )
    out = _sv_to_detections(dets, list(COCO80_CONTIGUOUS_LABELS))
    assert len(out) == 1
    assert out[0].class_id == 0
    assert out[0].label == "person"


def test_engine_empty_detections() -> None:
    dets = _FakeSvDetections(
        xyxy=np.zeros((0, 4)),
        confidence=np.zeros(0),
        class_id=np.zeros(0, dtype=int),
        data={},
    )
    out = _sv_to_detections(dets, list(COCO80_CONTIGUOUS_LABELS))
    assert out == []


# ---------------------------------------------------------------------------
# Evaluator integration: class-aware AP should NOT collapse after the fix
# ---------------------------------------------------------------------------


def test_remapped_predictions_match_gt_under_class_aware_evaluator() -> None:
    """Synthetic: GT class=person (contiguous 0), engine returns official id 1.

    Pre-fix the engine would label this 'bicycle' (contiguous[1]). Post-fix
    it labels 'person' (contiguous 0), so a class-aware evaluator at the
    contiguous level matches correctly.
    """
    from visionservex.runtime.evaluation import DetectionEvaluator

    ev = DetectionEvaluator()
    gt = [10.0, 10.0, 50.0, 50.0]
    # Build the remapped detection (engine layer would do this):
    cid, label, _src = remap_official_to_contiguous(1)
    ev.add_image(
        pred_boxes=[gt],
        pred_scores=[0.99],
        pred_classes=[label],
        gt_boxes=[gt],
        gt_classes=["person"],
    )
    m = ev.compute_metrics(iou_threshold=0.5)
    assert m["map50"] == 1.0  # post-remap, class-aware AP is perfect
    assert cid == 0
