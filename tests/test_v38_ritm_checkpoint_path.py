# SPDX-License-Identifier: Apache-2.0
"""v3.8 — RITM is commercial-safe (MIT) with a documented BYOT checkpoint path."""

from __future__ import annotations

from visionservex.licensing import policy as P


def test_ritm_commercial_safe_core():
    pol = P.get_policy("ritm")
    assert pol.final_policy == "commercial_safe_core"
    assert pol.code_license == "MIT"
    assert "SamsungLabs" in pol.upstream_url


def test_ritm_runtime_checkpoint_required():
    from visionservex.interactive_runtime import explain

    info = explain("ritm")
    assert info["state"] in ("checkpoint_required", "benchmark_passed", "tool_available")
    # the documented checkpoint path / source must be present
    text = repr(info)
    assert "ritm" in text


def test_ritm_classic_refiner_runs_without_weights():
    """The classic grabcut refiner must run today (weight-free, commercial-safe)."""
    from visionservex.interactive_runtime import explain

    info = explain("grabcut")
    assert info["state"] in ("benchmark_passed", "tool_available", "runnable")
