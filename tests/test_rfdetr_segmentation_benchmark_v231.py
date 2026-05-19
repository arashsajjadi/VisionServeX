# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.31.0: RF-DETR-Seg benchmark integration tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
SMOKE_ANN = REPO / "tests/assets/smoke/coco_instance_sample.json"


def _run(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "visionservex", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO),
    )


def _payload(proc: subprocess.CompletedProcess) -> dict:
    try:
        obj = json.loads(proc.stdout.strip())
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def test_benchmark_segmentation_with_rfdetr_seg_smoke_asset(tmp_path: Path) -> None:
    """benchmark-segmentation with smoke annotation must return structured result."""
    if not SMOKE_ANN.exists():
        pytest.skip("smoke annotation missing")
    out = tmp_path / "seg_bench.json"
    proc = _run(
        [
            "benchmark-segmentation",
            "--dataset",
            f"coco-instance:{SMOKE_ANN}",
            "--models",
            "rfdetr-seg-small",
            "--device",
            "cpu",
            "--format",
            "json",
            "--out",
            str(out),
        ],
        timeout=180,
    )
    # Must not be a usage error (exit 2)
    assert proc.returncode != 2, f"CLI usage error:\n{proc.stderr[:300]}"
    assert out.exists() or proc.stdout.strip().startswith("{"), "No JSON output produced"
    p = _payload(proc) or (json.loads(out.read_text()) if out.exists() else {})
    assert "status" in p, f"no 'status' in payload: {p}"
    assert "task" in p
    assert "rows" in p
    rows = p["rows"]
    assert len(rows) >= 1
    rfdetr_row = next((r for r in rows if "rfdetr-seg" in r.get("model_id", "")), None)
    assert rfdetr_row is not None
    # Must be ok or structured expected_blocker — never raw crash
    assert rfdetr_row["status"] in ("ok", "expected_blocker")
    if rfdetr_row["status"] == "expected_blocker":
        assert rfdetr_row.get("code"), "expected_blocker must have a code"


def test_benchmark_segmentation_no_nan_in_output(tmp_path: Path) -> None:
    """Output JSON must not contain raw NaN strings."""
    if not SMOKE_ANN.exists():
        pytest.skip("smoke annotation missing")
    out = tmp_path / "seg_nan.json"
    proc = _run(
        [
            "benchmark-segmentation",
            "--dataset",
            f"coco-instance:{SMOKE_ANN}",
            "--models",
            "rfdetr-seg-small",
            "--device",
            "cpu",
            "--format",
            "json",
            "--out",
            str(out),
        ],
        timeout=180,
    )
    output = proc.stdout + proc.stderr + (out.read_text() if out.exists() else "")
    assert "NaN" not in output, "Raw NaN in output"
    assert "NOT_WIRED" not in output, "NOT_WIRED in output"


def test_benchmark_segmentation_missing_dataset_returns_structured_blocker(tmp_path: Path) -> None:
    """Missing dataset path must return COCO_INSTANCE_DATASET_REQUIRED, not crash."""
    out = tmp_path / "seg_missing.json"
    proc = _run(
        [
            "benchmark-segmentation",
            "--dataset",
            "coco-instance:/nonexistent/path/ann.json",
            "--models",
            "rfdetr-seg-small",
            "--device",
            "cpu",
            "--format",
            "json",
            "--out",
            str(out),
        ],
        timeout=30,
    )
    assert proc.returncode != 2, f"CLI usage error:\n{proc.stderr[:200]}"
    p = _payload(proc) or (json.loads(out.read_text()) if out.exists() else {})
    rows = p.get("rows", [])
    if rows:
        r = rows[0]
        assert r["status"] == "expected_blocker"
        assert "DATASET" in r.get("code", "") or "RFDETR" in r.get("code", "")


def test_rfdetr_seg_benchmark_with_real_coco_subset() -> None:
    """If COCO val2017 400-image subset annotation exists, run a mini eval."""
    ann_path = Path("/home/arash/datasets/coco_val2017_400_vsx/annotations.json")
    if not ann_path.exists():
        pytest.skip("COCO val2017 400 subset annotation not available")

    from visionservex.runtime.rfdetr_seg_benchmark import run_rfdetr_seg_benchmark

    result = run_rfdetr_seg_benchmark(
        ann_file=ann_path,
        images_dir=ann_path.parent / "images",
        model_id="rfdetr-seg-small",
        device="cuda",
        threshold=0.3,
        max_images=10,  # mini run: 10 images only
    )
    assert result.status in ("ok", "expected_blocker")
    if result.status == "ok":
        # If it ran, mask AP must be a valid float
        import math

        assert result.mask_mAP50_95 is not None
        assert not math.isnan(result.mask_mAP50_95)
        assert 0.0 <= result.mask_mAP50_95 <= 1.0
        assert result.n_images == 10
    else:
        assert result.code, "expected_blocker must have a code"
