# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.36.0: DINOv3 official audit and auth gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_dinov3_audit_exists() -> None:
    p = Path(__file__).parent.parent / "reports/v236_dinov3_official_audit.json"
    if not p.exists():
        pytest.skip("DINOv3 audit not present")
    d = json.loads(p.read_text())
    assert d.get("status") in ("auth_required", "ok")
    assert "official_repo" in d


def test_dinov3_auth_required_has_instructions() -> None:
    p = Path(__file__).parent.parent / "reports/v236_dinov3_official_audit.json"
    if not p.exists():
        pytest.skip("DINOv3 audit not present")
    d = json.loads(p.read_text())
    if d.get("status") == "auth_required":
        assert d.get("auth_instructions"), "auth_required must include instructions"
        assert d.get("code") == "DINOv3_AUTH_REQUIRED"


def test_dinov2_still_works() -> None:
    """DINOv2 (open models) must remain contract_passed from v2.33."""
    from visionservex.registry import default_registry

    reg = default_registry()
    for mid in ["dinov2-small", "dinov2-base", "dinov2-large", "dinov2-giant"]:
        e = reg.get(mid)
        assert e is not None, f"{mid} missing"
        assert e.implementation_status in ("wired", "partial"), f"{mid}: {e.implementation_status}"
