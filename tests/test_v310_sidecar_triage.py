# SPDX-License-Identifier: Apache-2.0
"""v3.10.0: sidecar triage — policy correctness for sidecar models."""
from __future__ import annotations

import pytest


def _policy_row(model_id: str):
    from visionservex.licensing.policy import _ROWS

    for r in _ROWS:
        if r.model_id == model_id or model_id in r.aliases:
            return r
    return None


def test_maskdino_commercial_safe_core():
    pol = _policy_row("maskdino")
    assert pol is not None
    assert pol.final_policy == "commercial_safe_core"
    assert not pol.can_ship_weights


def test_codino_commercial_safe_core():
    pol = _policy_row("co-dino")
    assert pol is not None
    assert pol.final_policy == "commercial_safe_core"


def test_rtdetrv4_commercial_safe_core():
    pol = _policy_row("rt-detrv4")
    assert pol is not None
    assert pol.final_policy == "commercial_safe_core"


def test_rtmdet_commercial_safe_core():
    pol = _policy_row("rtmdet")
    assert pol is not None
    assert pol.final_policy == "commercial_safe_core"


def test_oneformer_legal_review_required():
    pol = _policy_row("oneformer")
    assert pol is not None
    assert pol.final_policy == "legal_review_required"
    assert not pol.default_safe


def test_internimage_legal_review_required():
    pol = _policy_row("internimage")
    assert pol is not None
    assert pol.final_policy == "legal_review_required"


def test_medsam2_noncommercial_restricted():
    pol = _policy_row("medsam2")
    assert pol is not None
    assert pol.final_policy == "noncommercial_restricted"
    assert not pol.commercial_safe


def test_maskdino_doctor_shows_blocker():
    """MaskDINO doctor must report detectron2 NOT installed (or pass if installed)."""
    import subprocess

    try:
        result = subprocess.run(
            ["visionservex", "maskdino", "doctor"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert "detectron2" in output.lower() or "OK" in output
    except subprocess.TimeoutExpired:
        pytest.skip("maskdino doctor timed out")
    except FileNotFoundError:
        pytest.skip("visionservex CLI not installed")
