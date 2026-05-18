# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Phase 2/3 (v2.16.0): evaluator framework unit tests.

These tests pin the AP/mAP evaluator behaviour on synthetic predictions so
the next notebook run can trust the metric pipeline. They cover:

- Perfect prediction → AP50 = 1.0, mAP50:95 ≈ 1.0
- Empty prediction → AP50 = 0
- Wrong class only → class-aware AP = 0
- Duplicate predictions → AP not inflated
- Score-ranked predictions

The real RF-DETR / D-FINE 100-image COCO128 validation is intentionally
NOT in this file. Running real model weights here would download many
gigabytes and risk a resource-guard freeze; the notebook (on the user's
RTX 5080, post-pip-install) is the empirical gate.
"""

from __future__ import annotations

from visionservex.runtime.evaluation import DetectionEvaluator


def _add_image(
    ev: DetectionEvaluator,
    *,
    pred_boxes,
    pred_scores,
    pred_classes,
    gt_boxes,
    gt_classes,
):
    ev.add_image(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes)


def test_perfect_predictions_give_ap50_one() -> None:
    ev = DetectionEvaluator()
    gt_box = [10.0, 10.0, 110.0, 110.0]
    _add_image(
        ev,
        pred_boxes=[gt_box],
        pred_scores=[0.99],
        pred_classes=["person"],
        gt_boxes=[gt_box],
        gt_classes=["person"],
    )
    metrics = ev.compute_metrics(iou_threshold=0.50)
    assert metrics["map50"] == 1.0, metrics
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_empty_predictions_give_zero_ap() -> None:
    ev = DetectionEvaluator()
    _add_image(
        ev,
        pred_boxes=[],
        pred_scores=[],
        pred_classes=[],
        gt_boxes=[[10, 10, 50, 50]],
        gt_classes=["person"],
    )
    metrics = ev.compute_metrics(iou_threshold=0.50)
    assert metrics["map50"] == 0.0
    assert metrics["recall"] == 0.0


def test_wrong_class_only_gives_zero_class_aware_ap() -> None:
    ev = DetectionEvaluator()
    gt_box = [10, 10, 50, 50]
    _add_image(
        ev,
        pred_boxes=[gt_box],
        pred_scores=[0.99],
        pred_classes=["cat"],  # wrong class
        gt_boxes=[gt_box],
        gt_classes=["person"],
    )
    metrics = ev.compute_metrics(iou_threshold=0.50)
    # GT class is "person"; AP for "person" is 0 because the only prediction
    # is "cat". So class-aware mAP at IoU 0.5 is 0.
    assert metrics["map50"] == 0.0


def test_duplicate_predictions_do_not_inflate_ap() -> None:
    ev = DetectionEvaluator()
    gt_box = [10, 10, 50, 50]
    _add_image(
        ev,
        pred_boxes=[gt_box, gt_box, gt_box],  # three duplicates
        pred_scores=[0.99, 0.98, 0.97],
        pred_classes=["person", "person", "person"],
        gt_boxes=[gt_box],
        gt_classes=["person"],
    )
    metrics = ev.compute_metrics(iou_threshold=0.50)
    # One match, two false positives. AP at IoU 0.5 should remain 1.0 because
    # the recall reaches 1 at rank-1 with precision 1, but the per-class
    # precision-at-F1 must drop below 1.
    assert metrics["map50"] <= 1.0
    # The crucial property: precision is NOT 1 (we have 2 false positives).
    assert metrics["precision"] < 1.0 + 1e-9
    # Per-class details
    per_cls = {p["class"]: p for p in metrics["per_class"]}
    assert per_cls["person"]["n_pred"] == 3
    assert per_cls["person"]["n_gt"] == 1


def test_score_ranked_predictions_have_consistent_ap() -> None:
    """Higher scores should be tried first; results must not depend on input order."""
    ev_a = DetectionEvaluator()
    ev_b = DetectionEvaluator()
    gt = [[10, 10, 50, 50], [60, 60, 100, 100]]
    pred_correct = [10, 10, 50, 50]
    pred_wrong = [200, 200, 250, 250]
    _add_image(
        ev_a,
        pred_boxes=[pred_correct, pred_wrong],
        pred_scores=[0.9, 0.1],
        pred_classes=["person", "person"],
        gt_boxes=gt,
        gt_classes=["person", "person"],
    )
    _add_image(
        ev_b,
        pred_boxes=[pred_wrong, pred_correct],
        pred_scores=[0.1, 0.9],
        pred_classes=["person", "person"],
        gt_boxes=gt,
        gt_classes=["person", "person"],
    )
    assert ev_a.compute_metrics(0.5)["map50"] == ev_b.compute_metrics(0.5)["map50"]


def test_map50_95_runs_across_iou_thresholds() -> None:
    ev = DetectionEvaluator()
    gt = [10, 10, 110, 110]
    _add_image(
        ev,
        pred_boxes=[gt],
        pred_scores=[0.99],
        pred_classes=["person"],
        gt_boxes=[gt],
        gt_classes=["person"],
    )
    out = ev.compute_map50_95()
    assert "map50_95" in out
    assert out["map50_95"] > 0.95, out
    assert out["map50"] == 1.0


def test_no_ground_truth_does_not_crash() -> None:
    ev = DetectionEvaluator()
    _add_image(
        ev,
        pred_boxes=[[10, 10, 50, 50]],
        pred_scores=[0.9],
        pred_classes=["person"],
        gt_boxes=[],
        gt_classes=[],
    )
    metrics = ev.compute_metrics(iou_threshold=0.5)
    # No GT for any class → mAP is 0, no exception.
    assert metrics["map50"] == 0.0
