# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Phase 12 (v2.16.0): public CLI equivalents for repo-local shell scripts.

The v16 notebook contract referenced ``scripts/run_anomaly_smoke.sh`` and
``scripts/run_video_search_smoke.sh``. In a PyPI/Colab install no
``scripts/`` directory exists, so the notebook crashed. v2.16.0 provides
notebook-safe CLI replacements that always emit structured JSON.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _vsx_cmd() -> list[str]:
    binary = shutil.which("visionservex")
    return [binary] if binary else [sys.executable, "-m", "visionservex"]


def _run(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(_vsx_cmd() + args, capture_output=True, text=True, timeout=timeout)


def test_anomaly_doctor_supports_format_and_out(tmp_path: Path) -> None:
    out = tmp_path / "anomaly_doctor.json"
    res = _run(["anomaly", "doctor", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["status"] in {"ok", "expected_blocker"}
    assert payload["code"] in {"OK", "ANOMALIB_REQUIRED"}
    assert "anomalib_installed" in payload


def test_anomaly_smoke_replaces_shell_script(tmp_path: Path) -> None:
    """`visionservex anomaly smoke` is the documented replacement for run_anomaly_smoke.sh."""
    out = tmp_path / "anomaly_smoke.json"
    res = _run(["anomaly", "smoke", "--model", "patchcore", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["status"] in {"ok", "expected_blocker"}
    assert "anomalib_installed" in payload


def test_video_search_smoke_returns_structured(tmp_path: Path) -> None:
    out = tmp_path / "video_search_smoke.json"
    res = _run(["video-search", "smoke", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["status"] in {"ok", "expected_blocker"}
    assert "trackers" in payload
    assert "reid" in payload


def test_dev_make_synthetic_video_help_lists_required_flags() -> None:
    res = _run(["dev", "make-synthetic-video", "--help"])
    assert res.returncode == 0
    assert "--out" in res.stdout
    assert "--frames" in res.stdout
