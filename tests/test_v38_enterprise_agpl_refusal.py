# SPDX-License-Identifier: Apache-2.0
"""v3.8 — AGPL / enterprise models are disabled in commercial-safe core."""

from __future__ import annotations

import pytest

from visionservex import hf_auth as H
from visionservex.licensing import policy as P

ENT = [r.model_id for r in P.iter_policies() if r.final_policy == "enterprise_license_required"]


def test_enterprise_present():
    assert "yolov8-seg" in ENT
    assert "fastsam-s" in ENT


@pytest.mark.parametrize("mid", ENT)
def test_enterprise_invariants(mid):
    pol = P.get_policy(mid)
    assert pol.default_safe is False
    assert pol.commercial_safe is False
    assert pol.production_allowed is False
    assert ("AGPL" in pol.warning_text) or ("enterprise" in pol.warning_text.lower())


def test_require_license_refuses_enterprise():
    with pytest.raises(H.HFLicenseError) as exc:
        H.hf_require_user_accepted_license("yolov8-seg")
    assert exc.value.state == "enterprise_license_required"


def test_download_blocked_for_enterprise():
    g = H.hf_download_allowed_by_policy("fastsam-x")
    assert g["allowed"] is False
    assert g["reason"] == "enterprise_or_agpl_license_required"


def test_no_enterprise_model_in_default_safe_core():
    for pol in P.iter_policies():
        if pol.default_safe:
            assert pol.final_policy == "commercial_safe_core"
            assert "AGPL" not in pol.weights_license
