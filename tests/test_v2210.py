# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.21.0 hardening tests.

Cover the failures surfaced by the v23 notebook run:
- DEIMv2 atto/femto/pico/n/l/x registry entries.
- result_classifier knows the v2.21 blocker codes
  (BYTETRACK_REQUIRED, TORCHREID_REQUIRED, OCSORT_REQUIRED,
  TOTAL_SEGMENTATOR_REQUIRED, NNUNET_REQUIRED, OPENMMLAB_REQUIRED, etc.).
- `model-zoo sources --family X --format json --out PATH`.
- Tracker alias normalisation (oc-sort, OC-SORT, byte-track → canonical).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _vsx() -> list[str]:
    binary = shutil.which("visionservex")
    return [binary] if binary else [sys.executable, "-m", "visionservex"]


def _run(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(_vsx() + args, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------------------
# DEIMv2 registry
# ---------------------------------------------------------------------------


def test_deimv2_all_eight_variants_present() -> None:
    from visionservex.registry import default_registry

    reg = default_registry()
    ids = {e.id for e in reg.list(task="detect")}
    for size in ("atto", "femto", "pico", "n", "s", "m", "l", "x"):
        assert f"deimv2-{size}" in ids, f"deimv2-{size} missing"


def test_deimv2_entries_are_structured_blockers() -> None:
    from visionservex.registry import default_registry

    reg = default_registry()
    for size in ("atto", "femto", "pico", "n", "s", "m", "l", "x"):
        entry = reg.get(f"deimv2-{size}")
        assert entry.implementation_status == "stub", entry.id
        assert entry.status == "experimental"
        assert entry.model_category == "experimental_sota"
        assert entry.unavailable_reason, entry.id


# ---------------------------------------------------------------------------
# result_classifier
# ---------------------------------------------------------------------------


def test_classifier_knows_all_v221_blocker_codes() -> None:
    from visionservex.runtime.result_classifier import EXPECTED_BLOCKER_CODES

    for code in (
        "BYTETRACK_REQUIRED",
        "TORCHREID_REQUIRED",
        "OCSORT_REQUIRED",
        "TOTAL_SEGMENTATOR_REQUIRED",
        "NNUNET_REQUIRED",
        "OPENMMLAB_REQUIRED",
        "DETECTRON2_REQUIRED",
        "MMDET_REQUIRED",
        "MEDSAM2_REQUIRED",
        "DEIM_REQUIRED",
        "DEIMV2_REQUIRED",
        "RTDETRV4_REQUIRED",
        "NON_CORE_LICENSE_OPT_IN_REQUIRED",
    ):
        assert code in EXPECTED_BLOCKER_CODES, code


def test_classifier_buckets_bytetrack_required_as_expected_blocker() -> None:
    from visionservex.runtime.result_classifier import classify_command_result

    payload = {"status": "expected_blocker", "code": "BYTETRACK_REQUIRED"}
    r = classify_command_result(returncode=0, stdout=json.dumps(payload), stderr="")
    assert r.status == "expected_blocker"


def test_classifier_buckets_ocsort_required_as_expected_blocker() -> None:
    from visionservex.runtime.result_classifier import classify_command_result

    payload = {"status": "expected_blocker", "code": "OCSORT_REQUIRED"}
    r = classify_command_result(returncode=0, stdout=json.dumps(payload), stderr="")
    assert r.status == "expected_blocker"


# ---------------------------------------------------------------------------
# model-zoo sources --family / --model
# ---------------------------------------------------------------------------


def test_model_zoo_sources_family_dfine(tmp_path: Path) -> None:
    out = tmp_path / "dfine.json"
    res = _run(["model-zoo", "sources", "--family", "dfine", "--format", "json", "--out", str(out)])
    assert res.returncode == 0, res.stderr
    payload = json.loads(out.read_text())
    assert isinstance(payload, list)
    assert payload
    assert all(e["family"].lower() == "dfine" for e in payload)


def test_model_zoo_sources_single_model(tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    res = _run(
        [
            "model-zoo",
            "sources",
            "--model",
            "dfine-s-o365-coco",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    payload = json.loads(out.read_text())
    assert len(payload) == 1
    assert payload[0]["model_id"] == "dfine-s-o365-coco"


def test_model_zoo_sources_help_lists_v221_flags() -> None:
    """v2.25.1: rich-aware help assertion."""
    from tests.helpers.cli_help import assert_help_contains_all, run_help

    res = run_help(["model-zoo", "sources"])
    assert_help_contains_all(res, ["--model", "--family", "--out", "--format"])


# ---------------------------------------------------------------------------
# Tracker alias normalisation
# ---------------------------------------------------------------------------


def test_tracker_alias_oc_sort_normalises_to_ocsort() -> None:
    from visionservex.runtime.trackers import TrackerUnavailableError, build_tracker

    # ocsort is not installed in test env, so this should raise
    # TrackerUnavailableError with code=OCSORT_REQUIRED — NOT TRACKER_UNKNOWN.
    try:
        build_tracker("oc-sort")
    except TrackerUnavailableError as exc:
        assert exc.code == "OCSORT_REQUIRED", exc.code
    else:
        # If ocsort IS installed, that's also fine — alias resolved correctly.
        pass


def test_tracker_alias_OC_SORT_uppercase_normalises() -> None:
    from visionservex.runtime.trackers import TrackerUnavailableError, build_tracker

    try:
        build_tracker("OC-SORT")
    except TrackerUnavailableError as exc:
        assert exc.code in ("OCSORT_REQUIRED", "TRACKER_REQUIRED")


def test_tracker_alias_bytetracker_normalises_to_bytetrack() -> None:
    from visionservex.runtime.trackers import TrackerUnavailableError, build_tracker

    try:
        build_tracker("bytetracker")
    except TrackerUnavailableError as exc:
        assert exc.code in ("BYTETRACK_REQUIRED", "TRACKER_REQUIRED")


def test_tracker_simple_iou_alias_still_returns_none() -> None:
    from visionservex.runtime.trackers import build_tracker

    assert build_tracker("simple_iou") is None
    assert build_tracker("simple-iou") is None
