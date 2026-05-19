# SPDX-License-Identifier: Apache-2.0
"""v2.33.0: every doctor command must return a structured JSON."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent

DOCTORS = [
    "all-benchmark",
    "detection",
    "segmentation",
    "promptable",
    "foundation",
    "dino",
    "sam3",
    "grounding-dino15",
    "florence2",
    "tracking",
    "anomaly",
    "openmmlab",
]


@pytest.mark.parametrize("name", DOCTORS)
def test_doctor_returns_structured_json(name: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "visionservex", "extra", name],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, f"{name}: returncode {proc.returncode}; stderr={proc.stderr[:200]}"
    # Output must be parseable JSON
    p = None
    try:
        p = json.loads(proc.stdout.strip())
    except Exception:
        # Try last JSON object
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    p = json.loads(line)
                    break
                except Exception:
                    continue
    assert isinstance(p, dict), f"{name}: not JSON"
    assert "status" in p
    assert "code" in p
    assert p["status"] in (
        "ok",
        "expected_blocker",
        "auth_required",
        "dependency_required",
        "sidecar_required",
    )


def test_doctor_sam3_returns_auth_required() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "visionservex", "extra", "sam3"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO),
    )
    p = json.loads(proc.stdout.strip())
    assert p.get("code") == "SAM3_AUTH_REQUIRED"
    assert "auth_instructions" in p


def test_doctor_grounding_dino15_returns_api_key_required() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "visionservex", "extra", "grounding-dino15"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO),
    )
    p = json.loads(proc.stdout.strip())
    assert "API" in p.get("code", "") or "AUTH" in p.get("code", "")
    assert "api_key_env_var" in p
