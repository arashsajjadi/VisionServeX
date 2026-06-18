# SPDX-License-Identifier: Apache-2.0
"""v3.19: every RF-DETR variant is TRAIN_READY_LIVE or has an exact blocker.

Weight-free — reads the committed RF-DETR live-train matrix + capabilities.
"""

from __future__ import annotations

import json
from pathlib import Path

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}
RFDETR = {m: c for m, c in CAPS.items() if c["family"] == "rfdetr"}
MATRIX = Path("docs/qa/v319_operationalize_all_models/rfdetr_live_train_matrix.json")


def test_rfdetr_family_exists():
    assert RFDETR


def test_each_rfdetr_is_train_live_or_blocked_with_reason():
    for mid, c in RFDETR.items():
        if c["readiness_state"] == taxonomy.TRAIN_READY_LIVE:
            assert c["live_verified_train"], mid
        else:
            # not train-live -> must carry an exact blocker (catalog-only seg-xl etc.)
            assert c["blocker"], mid


def test_committed_matrix_records_8_passes():
    if not MATRIX.exists():
        return
    data = json.loads(MATRIX.read_text())
    passed = {r["model_id"] for r in data["results"] if r["status"] == "PASS"}
    # The 8 trainable RF-DETR variants (5 detect + 3 seg) all passed live this sprint.
    expected = {
        "rfdetr-nano",
        "rfdetr-small",
        "rfdetr-medium",
        "rfdetr-base",
        "rfdetr-large",
        "rfdetr-seg-nano",
        "rfdetr-seg-small",
        "rfdetr-seg-medium",
    }
    assert expected <= passed, f"missing live RF-DETR passes: {expected - passed}"
    # every PASS row proves the full lifecycle
    for r in data["results"]:
        if r["status"] == "PASS":
            assert r["train"] and r["checkpoint_exists"] and r["reload"]
            assert r["predict_after_reload"] and r["output_schema_valid"]


def test_lifecycle_passes_are_train_ready_live_in_capabilities():
    if not MATRIX.exists():
        return
    data = json.loads(MATRIX.read_text())
    for r in data["results"]:
        if r["status"] == "PASS":
            assert CAPS[r["model_id"]]["readiness_state"] == taxonomy.TRAIN_READY_LIVE
