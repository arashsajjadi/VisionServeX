# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.45.0: verify the exact 51-model recovery plan exists and has all required models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PLAN_JSON = Path(__file__).resolve().parent.parent / "reports/v245_exact_51_recovery_plan.json"

REQUIRED_MODEL_IDS = frozenset({
    "deimv2-n", "bytetrack", "co-dino-inst-vit-l-coco", "co-dino-inst-vit-l-lvis",
    "edgesam", "internimage-b", "internimage-h", "internimage-l", "internimage-s",
    "internimage-t", "maskdino-r50-coco", "maskdino-r50-panoptic", "maskdino-swinl-coco",
    "medsam2", "oneformer-dinat-large", "rtmdet-r-l", "rtmdet-r-m", "rtmdet-r-s",
    "rtmdet-r-t", "rtmdet-r2-l", "rtmdet-r2-m", "rtmdet-r2-s", "rtmdet-r2-t",
    "seem-davit-d3", "seem-focal-t", "dino-x-api", "grounding-dino-1.5",
    "grounding-dino-1.5-pro", "grounding-dino-1.6", "grounding-dino-1.6-pro",
    "sam3-base", "fastsam-s", "fastsam-x", "prithvi-eo-2.0", "rfdetr-seg-2xlarge",
    "rfdetr-seg-xlarge", "totalsegmentator", "yolo-world", "yolo11l-seg.pt",
    "yolo11x-seg.pt", "yolo11x.pt", "yolo26x-seg.pt", "yolo26x.pt", "yolov10b.pt",
    "yolov8x-seg.pt", "yolov8x.pt", "agriclip", "deim-m", "deim-s",
    "dinov3-vitb16", "oneformer-convnext-large",
})


def test_recovery_plan_exists():
    assert PLAN_JSON.exists(), f"{PLAN_JSON} missing"


def test_recovery_plan_has_all_51_models():
    if not PLAN_JSON.exists():
        pytest.skip("recovery plan not present")
    data = json.loads(PLAN_JSON.read_text())
    found = {r["model_id"] for r in data.get("rows", [])}
    missing = REQUIRED_MODEL_IDS - found
    assert not missing, f"Missing from plan: {sorted(missing)}"


def test_recovery_plan_no_vague_actions():
    if not PLAN_JSON.exists():
        pytest.skip("recovery plan not present")
    data = json.loads(PLAN_JSON.read_text())
    bad = [
        r["model_id"]
        for r in data.get("rows", [])
        if not r.get("command_to_run") or not r.get("exact_current_problem")
    ]
    assert not bad, f"Rows with no command/problem: {bad[:5]}"


def test_recovery_plan_has_51_rows():
    if not PLAN_JSON.exists():
        pytest.skip("recovery plan not present")
    data = json.loads(PLAN_JSON.read_text())
    assert data["total"] == 51, f"Expected 51 rows, got {data['total']}"
