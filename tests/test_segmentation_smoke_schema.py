# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.29.0: segmentation smoke schema validation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SMOKE_IMG = Path("tests/assets/smoke/coco_person_car.jpg")
REPO_ROOT = Path(__file__).parent.parent


def _run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )


def _classify(proc: subprocess.CompletedProcess) -> tuple[str, dict | None]:
    import re

    from visionservex.runtime.result_classifier import classify_command_result

    cr = classify_command_result(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
    payload: dict | None = cr.structured_payload
    if payload is None:
        # Try full stdout (pretty-printed multi-line JSON)
        try:
            obj = json.loads(proc.stdout.strip())
            if isinstance(obj, dict):
                payload = obj
        except Exception:
            pass
    # Also try extracting first {...} block (handles leading log lines from rfdetr)
    if payload is None:
        m = re.search(r"\{.*\}", proc.stdout, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    payload = obj
            except Exception:
                pass
    if payload is None:
        for stream in (proc.stdout, proc.stderr):
            for line in stream.splitlines():
                s = line.strip()
                if s.startswith("{"):
                    try:
                        payload = json.loads(s)
                        break
                    except Exception:
                        pass
            if payload:
                break
    return cr.status, payload


# ---------------------------------------------------------------------------
# Auto segmentation smoke (rfdetr-seg-small via predict)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", ["rfdetr-seg-small", "rfdetr-seg-nano"])
def test_rfdetr_seg_smoke(model_id: str) -> None:
    """rfdetr-seg predict must return smoke_passed or expected_blocker."""
    if not SMOKE_IMG.exists():
        pytest.skip("smoke asset missing")
    proc = _run(
        [sys.executable, "-m", "visionservex", "predict", model_id, str(SMOKE_IMG), "--json"],
        timeout=90,
    )
    status, payload = _classify(proc)
    assert status in ("ok_clean", "ok_with_warning", "expected_blocker"), (
        f"{model_id}: unexpected classify status={status!r}\n"
        f"stdout={proc.stdout[:300]}\nstderr={proc.stderr[:300]}"
    )
    if status == "expected_blocker":
        assert payload is not None, f"{model_id}: expected_blocker but no structured payload"
        assert payload.get("code") or payload.get("status") == "expected_blocker"


# ---------------------------------------------------------------------------
# SAM2 smoke
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", ["sam2-hiera-tiny", "sam2.1-hiera-tiny"])
def test_sam2_smoke(model_id: str) -> None:
    """SAM2 predict with box prompt must return smoke_passed or expected_blocker."""
    if not SMOKE_IMG.exists():
        pytest.skip("smoke asset missing")
    proc = _run(
        [
            sys.executable,
            "-m",
            "visionservex",
            "predict",
            model_id,
            str(SMOKE_IMG),
            "--box",
            "50,50,200,200",
            "--json",
        ],
        timeout=90,
    )
    status, _payload = _classify(proc)
    assert status in ("ok_clean", "ok_with_warning", "expected_blocker"), (
        f"{model_id}: status={status!r}\nstdout={proc.stdout[:300]}"
    )


# ---------------------------------------------------------------------------
# Benchmark-segmentation command smoke (structured blocker)
# ---------------------------------------------------------------------------


def test_benchmark_segmentation_cmd_exists() -> None:
    """benchmark-segmentation command must be callable and return structured JSON."""
    proc = _run(
        [
            sys.executable,
            "-m",
            "visionservex",
            "benchmark-segmentation",
            "--dataset",
            "tests/assets/smoke/coco_instance_sample.json",
            "--models",
            "rfdetr-seg-small",
            "--out",
            "/tmp/vsx_seg_smoke_test.json",
            "--format",
            "json",
        ],
        timeout=30,
    )
    assert proc.returncode in (0, 1), (
        f"unexpected returncode {proc.returncode}\nstderr={proc.stderr[:200]}"
    )
    status, payload = _classify(proc)
    assert status != "failed_usage", (
        f"benchmark-segmentation: CLI usage error\nstderr={proc.stderr[:300]}"
    )
    # Must be structured
    assert payload is not None, (
        f"No structured payload from benchmark-segmentation\nstdout={proc.stdout[:300]}"
    )


# ---------------------------------------------------------------------------
# Promptable segmentation command smoke
# ---------------------------------------------------------------------------


def test_benchmark_promptable_segmentation_cmd_exists() -> None:
    """benchmark-promptable-segmentation must return structured JSON (not raw crash)."""
    proc = _run(
        [
            sys.executable,
            "-m",
            "visionservex",
            "benchmark-promptable-segmentation",
            "--dataset",
            "tests/assets/smoke/coco_instance_sample.json",
            "--images-dir",
            "tests/assets/smoke",
            "--models",
            "sam2-hiera-tiny",
            "--max-instances",
            "1",
            "--out",
            "/tmp/vsx_promptable_smoke_test.json",
            "--format",
            "json",
        ],
        timeout=90,
    )
    assert proc.returncode in (0, 1), (
        f"unexpected returncode {proc.returncode}\nstderr={proc.stderr[:200]}"
    )
    status, payload = _classify(proc)
    assert status != "failed_usage", (
        f"promptable-segmentation: CLI usage error\nstderr={proc.stderr[:300]}"
    )
    assert status in ("ok_clean", "ok_with_warning", "expected_blocker"), (
        f"unexpected status={status!r}\nstdout={proc.stdout[:300]}"
    )
    assert payload is not None, "No structured payload"
    assert payload.get("code") or payload.get("status")


def test_benchmark_promptable_max_instances_flag() -> None:
    """--max-instances flag (not --max-instances-per-image) must be accepted."""
    proc = _run(
        [
            sys.executable,
            "-m",
            "visionservex",
            "benchmark-promptable-segmentation",
            "--dataset",
            "tests/assets/smoke/coco_instance_sample.json",
            "--models",
            "sam2-hiera-tiny",
            "--max-instances",
            "3",
            "--out",
            "/tmp/vsx_promptable_flag_test.json",
            "--format",
            "json",
        ],
        timeout=30,
    )
    # Must NOT be a usage error
    assert proc.returncode != 2, (
        f"--max-instances flag not accepted (usage error)\nstderr={proc.stderr[:300]}"
    )
