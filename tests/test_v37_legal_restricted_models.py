# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7: restricted/non-commercial models must never enter the commercial-safe core."""
from __future__ import annotations

import csv
from pathlib import Path

R = Path(__file__).parent.parent / "notebook" / "99_final_report" / "reports"

# These must all be commercial_safe == False everywhere they appear.
RESTRICTED = ["edgesam", "fastsam-s", "fastsam-x", "yolov8-seg", "yolo11-seg",
              "locateanything-3b", "locate-anything-3b"]


def _inv():
    return {r["item_id"]: r for r in csv.DictReader((R / "v37_post_v259_inventory.csv").open())}


def test_restricted_never_commercial_safe():
    inv = _inv()
    for m in RESTRICTED:
        if m in inv:
            assert inv[m]["commercial_safe"] == "False", f"{m} must not be commercial_safe"
            assert inv[m]["default_safe"] == "False"


def test_edgesam_locateanything_excluded():
    inv = _inv()
    assert inv["edgesam"]["current_state"] == "excluded_restricted"
    assert inv["locateanything-3b"]["current_state"] == "excluded_restricted"


def test_ultralytics_agpl_excluded():
    inv = _inv()
    for m in ["yolov8-seg", "yolo11-seg"]:
        assert "AGPL" in inv[m]["license_status"]
        assert inv[m]["current_state"] == "excluded_restricted"


def test_locateanything_vsx_excluded():
    from visionservex.vsx import VSX
    h = VSX.locateanything("locate-anything-3b")
    info = h.explain()
    assert info["state"] == "excluded_restricted"
    assert info["commercial_safe"] is False
    assert info["default_safe"] is False


def test_license_decisions_consistency():
    """Every restricted model in the license CSV must be commercial_safe in {no,unclear}."""
    dec = {r["item_id"]: r for r in csv.DictReader((R / "v37_license_decisions.csv").open())}
    for m in ["edgesam", "yolov8-seg", "yolo11-seg", "locateanything-3b"]:
        if m in dec:
            assert dec[m]["commercial_safe"] in ("no", "unclear")
