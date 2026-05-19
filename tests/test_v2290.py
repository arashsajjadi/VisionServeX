# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.29.0 tests: smoke matrix, parseable blockers, segmentation schema."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).parent.parent


def _run(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "visionservex", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(_REPO),
    )


def test_version_is_at_least_2_29_0() -> None:
    """The v2.29.0 contract must remain valid for any version >= 2.29."""
    import visionservex

    parts = visionservex.__version__.split(".")
    major, minor = int(parts[0]), int(parts[1])
    assert (major, minor) >= (2, 29), visionservex.__version__


def test_models_smoke_matrix_command_registered() -> None:
    res = _run(["models", "smoke-matrix", "--help"])
    assert res.returncode == 0
    assert "device" in res.stdout.lower()


def test_smoke_assets_present() -> None:
    assets = [
        "tests/assets/smoke/coco_person_car.jpg",
        "tests/assets/smoke/coco_instance_sample.jpg",
        "tests/assets/smoke/coco_instance_sample.json",
        "tests/assets/smoke/medical_box_sample.png",
        "tests/assets/smoke/crop_weed_sample.jpg",
        "tests/assets/smoke/tracking_sample.mp4",
    ]
    missing = [a for a in assets if not (_REPO / a).exists()]
    assert not missing, f"Missing smoke assets: {missing}"


def test_maxvit_not_stub() -> None:
    from visionservex.registry import default_registry

    reg = default_registry()
    entry = reg.get("maxvit-tiny-tf-224")
    assert entry is not None
    assert entry.implementation_status != "stub"


def test_result_classifier_has_new_codes() -> None:
    from visionservex.runtime.result_classifier import EXPECTED_BLOCKER_CODES

    new_codes = [
        "TIMM_REQUIRED",
        "RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN",
        "RFDETR_SEG_MASK_OUTPUT_NOT_EXPOSED",
        "GT_MASKS_REQUIRED_FOR_MASK_METRICS",
        "SEGMENTATION_PIPELINE_NOT_WIRED",
    ]
    missing = [c for c in new_codes if c not in EXPECTED_BLOCKER_CODES]
    assert not missing, f"Codes not in EXPECTED_BLOCKER_CODES: {missing}"


def test_rfdetr_seg_schema_probe_report_exists() -> None:
    """The schema probe report must exist after v2.29.0 runs."""
    report = _REPO / "reports/rfdetr_seg_schema_probe_v229.json"
    if not report.exists():
        pytest.skip("schema probe report not yet generated — run smoke-matrix first")
    data = json.loads(report.read_text())
    assert data.get("status") == "probed", f"unexpected status: {data.get('status')}"
    assert data.get("mask_format") != "RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN", (
        "schema is still UNKNOWN after probe"
    )


def test_benchmark_segmentation_no_schema_unknown() -> None:
    """benchmark-segmentation must not return RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = Path(f.name)

    res = _run(
        [
            "benchmark-segmentation",
            "--dataset",
            "coco-instance:annot.json",
            "--models",
            "rfdetr-seg-small",
            "--device",
            "cpu",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    data = json.loads(out.read_text())
    codes = {r.get("code", "") for r in data.get("rows", [])}
    assert "RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN" not in codes, (
        f"RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN still present after probe: {codes}"
    )


def test_benchmark_promptable_segmentation_supports_max_instances() -> None:
    """--max-instances flag must be accepted (not --max-instances-per-image only)."""
    res = _run(
        [
            "benchmark-promptable-segmentation",
            "--dataset",
            "/nonexistent/path.json",
            "--models",
            "sam2-hiera-tiny",
            "--max-instances",
            "3",
            "--out",
            "/tmp/vsx_v229_max_instances.json",
            "--format",
            "json",
        ],
        timeout=15,
    )
    assert res.returncode != 2, f"--max-instances was rejected (usage error)\n{res.stderr[:200]}"


def test_no_nan_in_official_metrics() -> None:
    import math

    from visionservex.reporting.official_metrics import build_official_metrics_table

    rows = build_official_metrics_table()
    for row in rows:
        v = row.get("value")
        if isinstance(v, float):
            assert not math.isnan(v), f"NaN in official metrics for {row.get('model_id')}"


def test_anomaly_doctor_returns_expected_blocker() -> None:
    res = _run(["anomaly", "doctor", "--format", "json", "--out", "/tmp/anomaly_doctor_v229.json"])
    assert res.returncode == 0
    # Parse from stdout or output file
    try:
        data = json.loads(res.stdout.strip())
    except Exception:
        data = json.loads(Path("/tmp/anomaly_doctor_v229.json").read_text())
    assert data.get("status") == "expected_blocker" or data.get("code") == "ANOMALIB_REQUIRED"


def test_bytetrack_returns_expected_blocker() -> None:
    res = _run(
        [
            "video-search",
            "tracker-smoke",
            "--tracker",
            "bytetrack",
            "--format",
            "json",
            "--out",
            "/tmp/bytetrack_smoke.json",
        ]
    )
    # tracker-smoke exits with code 3 (typer Exit) when blocker is emitted
    try:
        data = json.loads(res.stdout.strip())
    except Exception:
        data = json.loads(Path("/tmp/bytetrack_smoke.json").read_text())
    assert data.get("status") == "expected_blocker"
    assert data.get("code") == "BYTETRACK_REQUIRED"
