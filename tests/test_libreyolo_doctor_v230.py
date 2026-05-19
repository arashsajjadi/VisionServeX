# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.30.0: libreyolo doctor must always return a structured result."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent


def _run(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
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
    for line in proc.stdout.splitlines():
        s = line.strip()
        if s.startswith("{"):
            try:
                return json.loads(s)
            except Exception:
                pass
    return {}


def test_libreyolo_doctor_help() -> None:
    proc = _run(["libreyolo", "doctor", "--help"])
    assert proc.returncode == 0


def test_libreyolo_doctor_returns_json() -> None:
    proc = _run(["libreyolo", "doctor", "--format", "json"])
    assert proc.returncode in (0, 1)
    p = _payload(proc)
    assert "status" in p
    assert "code" in p
    assert p["code"] in ("OK", "LIBREYOLO_REQUIRED")


def test_libreyolo_doctor_writes_out_file(tmp_path: Path) -> None:
    out = tmp_path / "doctor.json"
    proc = _run(["libreyolo", "doctor", "--format", "json", "--out", str(out)])
    assert proc.returncode in (0, 1)
    assert out.exists()
    data = json.loads(out.read_text())
    assert "status" in data


def test_libreyolo_doctor_no_unstructured_crash() -> None:
    """Even with libreyolo missing, doctor must return structured JSON, never crash raw."""
    proc = _run(["libreyolo", "doctor", "--format", "json"])
    assert "Traceback" not in proc.stderr
