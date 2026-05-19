# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.30.0: libreyolo license-audit must classify every family with a verdict."""

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


def test_license_audit_returns_families() -> None:
    proc = _run(["libreyolo", "license-audit", "--format", "json"])
    assert proc.returncode == 0
    p = _payload(proc)
    assert "rows" in p
    assert isinstance(p["rows"], list)
    assert len(p["rows"]) > 0


def test_yolox_is_apache_2_0() -> None:
    proc = _run(["libreyolo", "license-audit", "--format", "json"])
    p = _payload(proc)
    yolox_row = next((r for r in p.get("rows", []) if r["family"] == "yolox"), None)
    assert yolox_row is not None
    assert "Apache" in yolox_row.get("weight_license", "")
    assert yolox_row.get("auto_pull") is True


def test_yolonas_is_blocked() -> None:
    proc = _run(["libreyolo", "license-audit", "--format", "json"])
    p = _payload(proc)
    yolonas_row = next((r for r in p.get("rows", []) if r["family"] == "yolonas"), None)
    assert yolonas_row is not None
    assert yolonas_row.get("auto_pull") is False
    assert yolonas_row.get("license_risk") in ("non_commercial", "non-commercial")


def test_yolo9_is_gpl_opt_in() -> None:
    proc = _run(["libreyolo", "license-audit", "--format", "json"])
    p = _payload(proc)
    y9 = next((r for r in p.get("rows", []) if r["family"] == "yolo9"), None)
    assert y9 is not None
    assert y9.get("auto_pull") is False
    assert "GPL" in y9.get("weight_license", "")


def test_dfine_is_permissive_default_safe() -> None:
    proc = _run(["libreyolo", "license-audit", "--format", "json"])
    p = _payload(proc)
    df = next((r for r in p.get("rows", []) if r["family"] == "dfine"), None)
    assert df is not None
    assert df.get("auto_pull") is True
