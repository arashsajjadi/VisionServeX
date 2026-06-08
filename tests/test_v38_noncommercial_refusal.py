# SPDX-License-Identifier: Apache-2.0
"""v3.8 — non-commercial models refuse production; research-only is gated."""

from __future__ import annotations

import pytest

from visionservex import hf_auth as H
from visionservex.licensing import policy as P

NC = [r.model_id for r in P.iter_policies() if r.final_policy == "noncommercial_restricted"]


def test_noncommercial_present():
    assert "edge-sam" in NC
    assert "locate-anything-3b" in NC


@pytest.mark.parametrize("mid", NC)
def test_noncommercial_invariants(mid):
    pol = P.get_policy(mid)
    assert pol.production_allowed is False
    assert pol.default_safe is False
    assert pol.commercial_safe is False
    assert pol.can_ship_weights is False
    assert "non-commercial" in pol.warning_text.lower()


def test_require_license_refuses_noncommercial_by_default():
    with pytest.raises(H.HFLicenseError) as exc:
        H.hf_require_user_accepted_license("edge-sam")
    assert exc.value.state == "noncommercial_restricted"
    assert "--research-only" in exc.value.next_command


def test_require_license_allows_research_only():
    out = H.hf_require_user_accepted_license(
        "edge-sam", research_only=True, accept_noncommercial=True
    )
    assert out["allowed"] is True
    assert out["mode"] == "research_only"


def test_download_allowed_false_for_noncommercial():
    g = H.hf_download_allowed_by_policy("locate-anything-3b")
    assert g["allowed"] is False
    assert g["reason"] == "noncommercial_research_only"
