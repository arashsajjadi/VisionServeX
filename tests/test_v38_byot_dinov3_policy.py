# SPDX-License-Identifier: Apache-2.0
"""v3.8 — DINOv3 is BYOT custom-license (NOT Apache like DINOv2)."""

from __future__ import annotations

import pytest

from visionservex.licensing import policy as P

DINOV3 = [r.model_id for r in P.iter_policies() if r.family == "dinov3"]


def test_dinov3_variants_present():
    assert "dinov3-vitb16" in DINOV3
    assert "dinov3-convnext-tiny" in DINOV3
    assert len(DINOV3) >= 8  # v3.9 added vits16plus, vith16plus, vitl16-sat, vit7b16-sat


@pytest.mark.parametrize("mid", DINOV3)
def test_each_dinov3_is_byot(mid):
    pol = P.get_policy(mid)
    assert pol.final_policy == "byot_license_required"
    assert pol.gated is True
    assert pol.default_safe is False
    assert pol.can_ship_weights is False
    assert pol.hf_repo.startswith("facebook/dinov3-")
    assert "DINOv3 License" in pol.weights_license
    assert "Apache" not in pol.weights_license


def test_dinov2_is_commercial_safe_not_dinov3():
    # the easy-to-confuse control: DINOv2 stays commercial-safe core, Apache-2.0
    pol = P.get_policy("dinov2-base")
    assert pol.final_policy == "commercial_safe_core"
    assert pol.weights_license == "Apache-2.0"


def test_dinov3_embed_blocks_without_access(monkeypatch):
    from visionservex import byot_runtime as B

    monkeypatch.setattr(B._H, "hf_is_logged_in", lambda: False)
    res = B.dinov3_embed("dinov3-vits16", image=None)
    assert res["status"] == "blocked"
    assert res["state"] in ("auth_required", "auth_required_license_pending")
