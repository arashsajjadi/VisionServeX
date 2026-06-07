# SPDX-License-Identifier: Apache-2.0
"""v2.46 reconciler license-truth and alias corrections must move
incorrectly-restricted models out of non-healthy."""

from __future__ import annotations


def test_known_corrections_has_v246_license_truth_entries() -> None:
    from visionservex.reporting.v239_reconciler import KNOWN_CORRECTIONS

    for mid in ("agriclip", "prithvi-eo-2.0", "dinov3-vitb16"):
        assert mid in KNOWN_CORRECTIONS
        assert KNOWN_CORRECTIONS[mid]["final_state"] == "wired"
        assert KNOWN_CORRECTIONS[mid]["blocker_code"] == ""
        assert "v246_correction_reason" in KNOWN_CORRECTIONS[mid]


def test_known_corrections_has_v246_alias_entries() -> None:
    from visionservex.reporting.v239_reconciler import KNOWN_CORRECTIONS

    for mid in ("deim-m", "deim-s"):
        assert mid in KNOWN_CORRECTIONS
        assert KNOWN_CORRECTIONS[mid]["final_state"] == "wired"
        assert "alias" in KNOWN_CORRECTIONS[mid].get("v246_correction_reason", "")


def test_known_corrections_has_oneformer_remap() -> None:
    from visionservex.reporting.v239_reconciler import KNOWN_CORRECTIONS

    entry = KNOWN_CORRECTIONS["oneformer-convnext-large"]
    assert entry["final_state"] == "wired"
    assert "shi-labs" in entry["v246_correction_reason"]


def test_known_corrections_retain_agpl_gates() -> None:
    """v2.46 must NOT promote AGPL Ultralytics models to benchmark_passed."""
    from visionservex.reporting.v239_reconciler import KNOWN_CORRECTIONS

    for mid in (
        "fastsam-s",
        "fastsam-x",
        "yolo-world",
        "yolo11l-seg.pt",
        "yolo11x-seg.pt",
        "yolo11x.pt",
        "yolo26x-seg.pt",
        "yolo26x.pt",
        "yolov10b.pt",
        "yolov8x-seg.pt",
        "yolov8x.pt",
        "totalsegmentator",
    ):
        assert mid in KNOWN_CORRECTIONS, f"{mid} missing from KNOWN_CORRECTIONS"
        assert KNOWN_CORRECTIONS[mid]["final_state"] == "opt_in_license_required", (
            f"{mid} must stay license-gated, got {KNOWN_CORRECTIONS[mid]}"
        )


def test_wired_state_has_priority_above_sidecar_required() -> None:
    """v2.46 corrections rely on `wired` outranking sidecar_required (40)."""
    from visionservex.reporting.v239_reconciler import STATE_PRIORITY

    assert STATE_PRIORITY.get("wired", -1) > STATE_PRIORITY.get("sidecar_required", -1)
    assert STATE_PRIORITY.get("wired", -1) > STATE_PRIORITY.get("opt_in_license_required", -1)
