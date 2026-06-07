# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.6 LocateAnything CLI integration tests.

Tests the `visionservex locate-anything` sub-commands:
  list, status, explain, install, run (with and without --accept-noncommercial).
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


def test_locate_anything_help_does_not_crash() -> None:
    res = _run(["locate-anything", "--help"])
    assert res.returncode == 0
    assert "locate-anything" in res.stdout.lower() or "locateanything" in res.stdout.lower() or "LocateAnything" in res.stdout


def test_locate_anything_list_returns_ten_models(tmp_path: Path) -> None:
    out = tmp_path / "list.json"
    res = _run(["locate-anything", "list", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert "models" in payload
    assert len(payload["models"]) == 10


def test_locate_anything_list_all_have_noncommercial_license(tmp_path: Path) -> None:
    out = tmp_path / "list.json"
    _run(["locate-anything", "list", "--format", "json", "--out", str(out)])
    payload = json.loads(out.read_text())
    for m in payload["models"]:
        assert m["default_safe"] is False
        assert m["commercial_safe"] is False
        assert "NVIDIA" in m["license"]


def test_locate_anything_status_returns_legal_review(tmp_path: Path) -> None:
    out = tmp_path / "status.json"
    res = _run(["locate-anything", "status", "locate-anything-3b", "--out", str(out)])
    assert res.returncode == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["state"] == "excluded_restricted"


def test_locate_anything_explain_has_warning(tmp_path: Path) -> None:
    out = tmp_path / "explain.json"
    _run(["locate-anything", "explain", "locate-anything-3b", "--out", str(out)])
    payload = json.loads(out.read_text())
    assert "NVIDIA" in payload["warning"]
    assert "non-commercial" in payload["warning"]


def test_locate_anything_install_prints_sidecar() -> None:
    res = _run(["locate-anything", "install"])
    assert res.returncode == 0
    output = res.stdout + res.stderr
    assert "eagle" in output.lower() or "Eagle" in output


def test_locate_anything_run_without_flag_returns_blocker(tmp_path: Path) -> None:
    """run without --accept-noncommercial must return NONCOMMERCIAL_ACKNOWLEDGMENT_REQUIRED."""
    out = tmp_path / "run.json"
    from PIL import Image as _PIL

    img = tmp_path / "img.jpg"
    _PIL.new("RGB", (64, 64)).save(img)
    res = _run(
        [
            "locate-anything", "run",
            "locate-anything-3b",
            str(img),
            "--text", "cat",
            "--out", str(out),
        ]
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["status"] == "expected_blocker"
    assert payload["code"] == "NONCOMMERCIAL_ACKNOWLEDGMENT_REQUIRED"


def test_locate_anything_run_warning_printed_to_stderr(tmp_path: Path) -> None:
    """The NVIDIA warning must appear in stderr even when flag is missing."""
    out = tmp_path / "run.json"
    from PIL import Image as _PIL

    img = tmp_path / "img.jpg"
    _PIL.new("RGB", (64, 64)).save(img)
    res = _run(
        [
            "locate-anything", "run",
            "locate-anything-3b",
            str(img),
            "--text", "dog",
            "--out", str(out),
        ]
    )
    assert "WARNING" in res.stderr or "NVIDIA" in res.stderr


def test_locate_anything_run_with_flag_attempts_sidecar(tmp_path: Path) -> None:
    """With --accept-noncommercial, should attempt sidecar; expected_blocker if not installed."""
    out = tmp_path / "run.json"
    from PIL import Image as _PIL

    img = tmp_path / "img.jpg"
    _PIL.new("RGB", (64, 64)).save(img)
    res = _run(
        [
            "locate-anything", "run",
            "locate-anything-3b",
            str(img),
            "--text", "cat",
            "--accept-noncommercial",
            "--out", str(out),
        ]
    )
    assert res.returncode == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    # Either ran (if sidecar installed) or returned sidecar_required blocker
    assert payload["status"] in {"ok", "expected_blocker"}
    if payload["status"] == "expected_blocker":
        assert payload["code"] in {"SIDECAR_REQUIRED", "NONCOMMERCIAL_ACKNOWLEDGMENT_REQUIRED"}
