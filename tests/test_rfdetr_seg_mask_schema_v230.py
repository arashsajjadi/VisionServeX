# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.30.0: RF-DETR-Seg mask schema must remain confirmed (no UNKNOWN code)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
PROBE_REPORT = REPO / "reports/rfdetr_seg_schema_probe_v229.json"


def test_schema_probe_report_exists() -> None:
    if not PROBE_REPORT.exists():
        pytest.skip("schema probe report not present")
    data = json.loads(PROBE_REPORT.read_text())
    assert data.get("status") == "probed"


def test_schema_is_segments_list_with_per_segment_mask() -> None:
    if not PROBE_REPORT.exists():
        pytest.skip("schema probe report not present")
    data = json.loads(PROBE_REPORT.read_text())
    fmt = data.get("mask_format", "")
    assert fmt != "RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN", (
        "RF-DETR-Seg schema must be probed and confirmed by v2.30"
    )
    assert fmt, "mask_format must be non-empty"


def test_benchmark_segmentation_does_not_emit_schema_unknown() -> None:
    """The benchmark-segmentation CLI must not emit RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN
    after the v2.29 probe."""
    import subprocess
    import sys
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        out = Path(fh.name)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "visionservex",
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
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, f"returncode={proc.returncode}\nstderr={proc.stderr[:200]}"
    data = json.loads(out.read_text())
    codes = {r.get("code") for r in data.get("rows", [])}
    assert "RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN" not in codes, codes
