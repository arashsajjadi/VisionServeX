# SPDX-License-Identifier: Apache-2.0
"""v3.8 — SAM 3 / SAM 3.1 are BYOT, gated, never default-safe, never shipped."""

from __future__ import annotations

import pytest

from visionservex.licensing import policy as P

SAM3 = [r.model_id for r in P.iter_policies() if r.family in ("sam3", "sam3.1")]


def test_sam3_family_present():
    assert any(m.startswith("sam3-") for m in SAM3)
    assert any(m.startswith("sam3.1-") for m in SAM3)


@pytest.mark.parametrize("mid", SAM3)
def test_each_sam3_is_byot(mid):
    pol = P.get_policy(mid)
    assert pol.final_policy == "byot_license_required"
    assert pol.gated is True
    assert pol.local_token_required is True
    assert pol.user_license_required is True
    assert pol.default_safe is False
    assert pol.commercial_safe is False
    assert pol.production_allowed is False
    assert pol.can_ship_weights is False
    assert pol.can_auto_download is False
    assert pol.hf_repo.startswith("facebook/sam3")
    assert "redistribute" in pol.warning_text.lower()


def test_sam3_custom_license_not_apache():
    pol = P.get_policy("sam3-base")
    assert "SAM License" in pol.weights_license
    assert "Apache" not in pol.weights_license


def test_sam3_segment_requires_text_or_blocks(monkeypatch):
    # Without access this must block (auth), never fabricate a mask.
    from visionservex import byot_runtime as B

    monkeypatch.setattr(B._H, "hf_is_logged_in", lambda: False)
    res = B.sam3_segment("sam3-base", image=None, text="person")
    assert res["status"] == "blocked"
    assert res["state"] in ("auth_required", "auth_required_license_pending")
