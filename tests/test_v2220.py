# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.22.0: DEIMv2 / RT-DETRv4 doctor + structured blocker tests.

The v2.22.0 release attempts real integration of DEIMv2 and RT-DETRv4 and
documents the EXACT upstream blockers found:

- DEIMv2 needs ``torch==2.5.1`` (strict pin) and the upstream Python
  package (not on PyPI; requires `git clone Intellindust-AI-Lab/DEIMv2`).
  The current VisionServeX install runs torch 2.11.0+cu130, so the
  blocker is the combined `TORCH_VERSION_CONFLICT` +
  `NEEDS_UPSTREAM_REPO`.
- RT-DETRv4 is **not yet released upstream**. The canonical
  `lyuwenyu/RT-DETR` repo ships rtdetr_pytorch/ and rtdetrv2_pytorch/
  only — no rtdetrv4_pytorch/ directory or v4 release tag as of
  2026-05-18. The blocker is `RTDETRV4_UPSTREAM_NOT_RELEASED`.

These tests pin the structured-blocker contract.
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
# DEIMv2
# ---------------------------------------------------------------------------


def test_deimv2_help_lists_subcommands() -> None:
    res = _run(["deimv2", "--help"])
    assert res.returncode == 0
    for sub in ("doctor", "pull", "smoke-test"):
        assert sub in res.stdout, sub


def test_deimv2_doctor_returns_structured_blocker(tmp_path: Path) -> None:
    """torch 2.11.0+cu130 vs pinned 2.5.1 + no upstream deimv2 module → expected_blocker."""
    out = tmp_path / "doctor.json"
    res = _run(["deimv2", "doctor", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    payload = json.loads(out.read_text())
    # If both blockers are present (the realistic test-env case), status is expected_blocker.
    # If a future env actually has DEIMv2 set up, status=ok is also acceptable.
    assert payload["status"] in {"expected_blocker", "ok"}
    if payload["status"] == "expected_blocker":
        assert payload["code"] == "DEIMV2_NOT_RUNNABLE"
        # Must include the structured blocker codes the v2.22 release documented.
        assert (
            "TORCH_VERSION_CONFLICT" in payload["blockers"]
            or "NEEDS_UPSTREAM_REPO" in payload["blockers"]
        )


def test_deimv2_pull_unpublished_variant_returns_checkpoint_not_found(tmp_path: Path) -> None:
    out = tmp_path / "pull.json"
    res = _run(["deimv2", "pull", "deimv2-x", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    payload = json.loads(out.read_text())
    assert payload["status"] == "expected_blocker"
    assert payload["code"] == "CHECKPOINT_NOT_FOUND"


def test_deimv2_smoke_test_returns_structured_blocker(tmp_path: Path) -> None:
    """smoke-test against missing upstream package must structured-block, not crash."""
    from PIL import Image as _PIL

    img = tmp_path / "img.jpg"
    _PIL.new("RGB", (32, 32)).save(img)
    out = tmp_path / "smoke.json"
    res = _run(
        [
            "deimv2",
            "smoke-test",
            "deimv2-s",
            str(img),
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    # Expected_blocker exit 0 (matches sam-family validate / anomaly doctor pattern).
    assert res.returncode == 0
    payload = json.loads(out.read_text())
    assert payload["status"] in {"expected_blocker", "ok"}
    if payload["status"] == "expected_blocker":
        assert payload["code"] == "DEIMV2_NOT_RUNNABLE"


# ---------------------------------------------------------------------------
# RT-DETRv4
# ---------------------------------------------------------------------------


def test_rtdetrv4_help_lists_subcommands() -> None:
    res = _run(["rtdetrv4", "--help"])
    assert res.returncode == 0
    for sub in ("doctor", "pull", "smoke-test"):
        assert sub in res.stdout, sub


def test_rtdetrv4_doctor_returns_upstream_not_released(tmp_path: Path) -> None:
    out = tmp_path / "doctor.json"
    res = _run(["rtdetrv4", "doctor", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    payload = json.loads(out.read_text())
    assert payload["status"] == "expected_blocker"
    assert payload["code"] == "RTDETRV4_UPSTREAM_NOT_RELEASED"
    # Evidence carries the upstream variants that ARE available.
    available = payload.get("evidence", {}).get("upstream_available_variants", [])
    assert "rtdetrv2_pytorch" in available
    assert "rtdetrv4_pytorch" not in available


def test_rtdetrv4_pull_returns_structured_blocker(tmp_path: Path) -> None:
    out = tmp_path / "pull.json"
    res = _run(["rtdetrv4", "pull", "rtdetrv4-s", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    payload = json.loads(out.read_text())
    assert payload["status"] == "expected_blocker"
    assert payload["code"] == "RTDETRV4_UPSTREAM_NOT_RELEASED"


def test_rtdetrv4_smoke_test_returns_structured_blocker(tmp_path: Path) -> None:
    from PIL import Image as _PIL

    img = tmp_path / "img.jpg"
    _PIL.new("RGB", (32, 32)).save(img)
    out = tmp_path / "smoke.json"
    res = _run(
        [
            "rtdetrv4",
            "smoke-test",
            "rtdetrv4-s",
            str(img),
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    payload = json.loads(out.read_text())
    assert payload["status"] == "expected_blocker"
    assert payload["code"] == "RTDETRV4_UPSTREAM_NOT_RELEASED"


# ---------------------------------------------------------------------------
# result_classifier knows the new v2.22 blocker codes
# ---------------------------------------------------------------------------


def test_classifier_knows_v222_blocker_codes() -> None:
    from visionservex.runtime.result_classifier import EXPECTED_BLOCKER_CODES

    for code in (
        "DEIMV2_NOT_RUNNABLE",
        "TORCH_VERSION_CONFLICT",
        "NEEDS_UPSTREAM_REPO",
        "HUGGINGFACE_HUB_REQUIRED",
        "CHECKPOINT_NOT_FOUND",
        "RTDETRV4_UPSTREAM_NOT_RELEASED",
        "DEPENDENCY_CONFLICT",
    ):
        assert code in EXPECTED_BLOCKER_CODES, code


def test_classifier_buckets_deimv2_not_runnable_as_expected_blocker() -> None:
    from visionservex.runtime.result_classifier import classify_command_result

    payload = {
        "status": "expected_blocker",
        "code": "DEIMV2_NOT_RUNNABLE",
        "blockers": ["TORCH_VERSION_CONFLICT", "NEEDS_UPSTREAM_REPO"],
    }
    r = classify_command_result(returncode=0, stdout=json.dumps(payload), stderr="")
    assert r.status == "expected_blocker"


def test_classifier_buckets_rtdetrv4_upstream_not_released() -> None:
    from visionservex.runtime.result_classifier import classify_command_result

    payload = {"status": "expected_blocker", "code": "RTDETRV4_UPSTREAM_NOT_RELEASED"}
    r = classify_command_result(returncode=0, stdout=json.dumps(payload), stderr="")
    assert r.status == "expected_blocker"
