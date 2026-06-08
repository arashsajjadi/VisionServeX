# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7 Priority-Zero table remainders: interactive seg, RF-DETR-Seg, GD+SAM
pipelines, SAM2.1 ONNX, tiny/HQ, edge/fast/ultralytics — each decided + wired."""

from __future__ import annotations

import csv
from pathlib import Path

R = Path(__file__).parent.parent / "notebook" / "99_final_report" / "reports"


def _inv():
    return {r["item_id"]: r for r in csv.DictReader((R / "v37_post_v259_inventory.csv").open())}


# P0-A interactive
def test_interactive_models_present_and_decided():
    inv = _inv()
    for m in ["ritm", "clickseg", "simpleclick", "focalclick"]:
        assert m in inv, f"{m} missing from inventory"
        assert inv[m]["current_state"] in ("checkpoint_required", "legal_review_required")


def test_ritm_is_commercial_candidate():
    inv = _inv()
    assert inv["ritm"]["product_grade_status"] in ("product_grade_candidate", "checkpoint_required")
    # ritm is the only commercial-safe deep interactive model
    assert inv["simpleclick"]["commercial_safe"] == "False"
    assert inv["focalclick"]["commercial_safe"] == "False"


# P0-B rfdetr-seg
def test_all_six_rfdetr_seg_present():
    inv = _inv()
    for v in ["nano", "small", "medium", "large", "xl", "2xl"]:
        assert f"rfdetr-seg-{v}" in inv


def test_rfdetr_seg_executed_three_plus():
    led = list(csv.DictReader((R / "v37_new_model_execution_ledger.csv").open()))
    seg_ok = [r for r in led if r["task"].startswith("rfdetrseg:") and r["status"] == "ok"]
    assert len(seg_ok) >= 3, f"need >=3 executed rfdetr-seg variants, got {len(seg_ok)}"


# P0-C pipelines
def test_pipelines_executed():
    led = list(csv.DictReader((R / "v37_new_model_execution_ledger.csv").open()))
    pipes = [r for r in led if r["task"].startswith("pipe:") and r["status"] == "ok"]
    assert len(pipes) >= 5, f"need >=5 executed GD+SAM pipelines, got {len(pipes)}"


# P0-D sam2.1 onnx
def test_sam21_onnx_documented():
    sam = {r["variant_id"]: r for r in csv.DictReader((R / "v37_sam_variant_matrix.csv").open())}
    for v in [
        "sam2.1-onnx-tiny",
        "sam2.1-onnx-small",
        "sam2.1-onnx-base-plus",
        "sam2.1-onnx-large",
    ]:
        assert v in sam
        assert sam[v]["final_state"] in ("blocked_documented", "benchmark_passed")


# P0-E tiny/hq
def test_tiny_hq_decided():
    sam = {r["variant_id"]: r for r in csv.DictReader((R / "v37_sam_variant_matrix.csv").open())}
    assert sam["hq-sam"]["final_state"] == "legal_review_required"
    assert sam["tinysam"]["final_state"] in ("checkpoint_required", "benchmark_passed")


# P0-F edge/fast/ultralytics
def test_restricted_excluded_from_core():
    inv = _inv()
    for m in ["edgesam", "yolov8-seg", "yolo11-seg"]:
        assert inv[m]["current_state"] == "excluded_restricted"
        assert inv[m]["commercial_safe"] == "False"
    for m in ["fastsam-s", "fastsam-x"]:
        assert inv[m]["commercial_safe"] == "False"
