# SPDX-License-Identifier: Apache-2.0
"""v3.11: INSID3 policy rows — family, license, BYOT invariants."""

from __future__ import annotations

import pytest


def _get_pol(mid):
    from visionservex.licensing.policy import get_policy, resolve_model_id

    return get_policy(resolve_model_id(mid))


@pytest.mark.parametrize("mid", ["insid3-small", "insid3-base", "insid3-large"])
def test_insid3_policy_exists(mid):
    pol = _get_pol(mid)
    assert pol is not None, f"Policy row missing for {mid}"


@pytest.mark.parametrize("mid", ["insid3-small", "insid3-base", "insid3-large"])
def test_insid3_family(mid):
    pol = _get_pol(mid)
    assert pol.family == "insid3"


@pytest.mark.parametrize("mid", ["insid3-small", "insid3-base", "insid3-large"])
def test_insid3_policy_is_byot(mid):
    pol = _get_pol(mid)
    assert pol.final_policy == "byot_license_required"


@pytest.mark.parametrize("mid", ["insid3-small", "insid3-base", "insid3-large"])
def test_insid3_cannot_ship_weights(mid):
    pol = _get_pol(mid)
    assert not pol.can_ship_weights, f"can_ship_weights must be False for {mid}"


@pytest.mark.parametrize("mid", ["insid3-small", "insid3-base", "insid3-large"])
def test_insid3_not_default_safe(mid):
    pol = _get_pol(mid)
    assert not pol.default_safe


@pytest.mark.parametrize("mid", ["insid3-small", "insid3-base", "insid3-large"])
def test_insid3_hf_repo_is_dinov3(mid):
    pol = _get_pol(mid)
    assert "facebook/dinov3" in pol.hf_repo, (
        f"INSID3 {mid} must point to a DINOv3 HF repo, got {pol.hf_repo}"
    )


@pytest.mark.parametrize("mid", ["insid3-small", "insid3-base", "insid3-large"])
def test_insid3_notes_attribution(mid):
    pol = _get_pol(mid)
    assert "Built with DINOv3" in (pol.notes or ""), (
        f"Policy notes for {mid} must include 'Built with DINOv3' attribution"
    )


def test_insid3_aliases():
    from visionservex.licensing.policy import resolve_model_id

    assert resolve_model_id("insid3") == "insid3-large"
    assert resolve_model_id("insid3-default") == "insid3-large"
    assert resolve_model_id("insid3-dinov3-large") == "insid3-large"
    assert resolve_model_id("insid3-dinov3-small") == "insid3-small"
    assert resolve_model_id("insid3-dinov3-base") == "insid3-base"


def test_insid3_rows_in_global_policy():
    from visionservex.licensing.policy import POLICY

    insid3_rows = [k for k in POLICY if POLICY[k].family == "insid3"]
    assert len(insid3_rows) == 3, f"Expected 3 INSID3 policy rows, got {len(insid3_rows)}"


def test_no_duplicate_model_ids():
    from visionservex.licensing.policy import POLICY

    seen = {}
    for mid, pol in POLICY.items():
        if pol.family == "insid3":
            assert mid not in seen, f"Duplicate INSID3 model_id: {mid}"
            seen[mid] = True
