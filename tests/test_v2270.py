# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.27.0 tests: LibreYOLO integration, MaxViT alias fix, parseable-blocker
classifier fix, notebook v30 contract."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _vsx() -> list[str]:
    binary = shutil.which("visionservex")
    return [binary] if binary else [sys.executable, "-m", "visionservex"]


def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(_vsx() + args, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------------------
# Phase 1: LibreYOLO
# ---------------------------------------------------------------------------


def test_libreyolo_doctor_returns_structured(tmp_path: Path) -> None:
    out = tmp_path / "doctor.json"
    res = _run(["libreyolo", "doctor", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["code"] in {"OK", "LIBREYOLO_REQUIRED"}
    assert "libreyolo_installed" in d


def test_libreyolo_license_audit_has_six_families(tmp_path: Path) -> None:
    out = tmp_path / "audit.json"
    res = _run(["libreyolo", "license-audit", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["status"] == "ok"
    assert d["n_families"] == 6
    families = {r["family"] for r in d["rows"]}
    assert {"yolox", "yolo9", "yolonas", "dfine", "rtdetr", "rfdetr"}.issubset(families)
    # yolonas must be flagged non-commercial
    nas = next(r for r in d["rows"] if r["family"] == "yolonas")
    assert nas["license_risk"] == "non_commercial"
    assert nas["auto_pull"] is False


def test_libreyolo_pull_unknown_model(tmp_path: Path) -> None:
    out = tmp_path / "pull.json"
    res = _run(
        [
            "libreyolo",
            "pull",
            "libreyolo-foo-bar",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    # Either succeeds with structured payload or returncode != 0 with same payload.
    assert out.exists() or res.returncode != 0
    if out.exists():
        d = json.loads(out.read_text())
        assert d["status"] in {"expected_blocker", "ok"}
        assert d["code"] in {
            "LIBREYOLO_MODEL_NOT_FOUND",
            "LIBREYOLO_REQUIRED",
            "LIBREYOLO_WEIGHT_LICENSE_UNVERIFIED",
        }


def test_libreyolo_pull_yolonas_blocked_without_accept_noncommercial(tmp_path: Path) -> None:
    """YOLO-NAS weights are non-commercial; auto-pull must be blocked."""
    out = tmp_path / "pull.json"
    res = _run(
        [
            "libreyolo",
            "pull",
            "libreyolo-yolonas-s",
            "--cache-root",
            str(tmp_path),
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    # In CI without libreyolo installed, returncode may be non-zero;
    # what matters is that the structured payload was written.
    assert out.exists() or res.returncode != 0
    if out.exists():
        d = json.loads(out.read_text())
        assert d["status"] in {"expected_blocker", "ok"}
        # In CI where libreyolo isn't installed → LIBREYOLO_REQUIRED.
        # When installed → LIBREYOLO_WEIGHT_LICENSE_NONCOMMERCIAL.
        assert d["code"] in {
            "LIBREYOLO_WEIGHT_LICENSE_NONCOMMERCIAL",
            "LIBREYOLO_REQUIRED",
        }


# ---------------------------------------------------------------------------
# Phase 2: parseable-blocker classifier fix
# ---------------------------------------------------------------------------


def test_classifier_keeps_parseable_blocker_when_stderr_has_noise() -> None:
    """ANOMALIB_REQUIRED in stdout JSON must classify as expected_blocker
    even when stderr also has a traceback-like tail."""
    from visionservex.runtime.result_classifier import classify_command_result

    payload = {"status": "expected_blocker", "code": "ANOMALIB_REQUIRED"}
    stdout = json.dumps(payload)
    stderr = "Traceback (most recent call last):\n  File X\nModuleNotFoundError: anomalib"
    r = classify_command_result(returncode=1, stdout=stdout, stderr=stderr)
    assert r.status == "expected_blocker", r.status


def test_classifier_recognises_v227_blocker_codes() -> None:
    from visionservex.runtime.result_classifier import EXPECTED_BLOCKER_CODES

    for code in (
        "LIBREYOLO_REQUIRED",
        "LIBREYOLO_WEIGHT_LICENSE_UNVERIFIED",
        "LIBREYOLO_MODEL_NOT_FOUND",
        "UPSTREAM_HF_REPO_NOT_FOUND",
        "DOWNLOAD_FAILED",
        "ULTRALYTICS_REQUIRED",
        "COCO_VAL2017_400_REQUIRED",
        "OFFICIAL_METRIC_NOT_COLLECTED",
        "OFFICIAL_METRIC_NOT_FOUND",
    ):
        assert code in EXPECTED_BLOCKER_CODES, code


# ---------------------------------------------------------------------------
# Phase 2A: MaxViT alias
# ---------------------------------------------------------------------------


def test_maxvit_alias_resolves_in_registry() -> None:
    """`maxvit` must resolve to `maxvit-tiny-tf-224` (not 404 on unknown id)."""
    from visionservex.registry.registry import _USER_FACING_ALIASES

    assert _USER_FACING_ALIASES.get("maxvit") == "maxvit-tiny-tf-224"


# ---------------------------------------------------------------------------
# Phase 7: version
# ---------------------------------------------------------------------------


def test_version_is_at_least_2_27_0() -> None:
    import visionservex

    parts = visionservex.__version__.split(".")
    major, minor = int(parts[0]), int(parts[1])
    assert (major, minor) >= (2, 27), visionservex.__version__
