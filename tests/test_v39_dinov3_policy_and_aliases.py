# SPDX-License-Identifier: Apache-2.0
"""v3.9 — DINOv3 policy: correct bucket, BYOT flags, no-ship invariant, aliases."""

from __future__ import annotations

import pytest

DINOV3_MODEL_IDS = [
    "dinov3-vits16",
    "dinov3-vitb16",
    "dinov3-vitl16",
    "dinov3-convnext-tiny",
    "dinov3-convnext-small",
    "dinov3-convnext-base",
    "dinov3-convnext-large",
    "dinov3-vith16plus",
    "dinov3-vit7b16",
    "dinov3-vits16plus",
]


@pytest.mark.parametrize("model_id", DINOV3_MODEL_IDS)
def test_dinov3_policy_bucket(model_id):
    from visionservex.licensing import policy as P

    pol = P.get_policy(model_id)
    assert pol is not None, f"No policy for {model_id}"
    assert pol.final_policy == "byot_license_required", (
        f"{model_id} expected byot_license_required, got {pol.final_policy}"
    )


@pytest.mark.parametrize("model_id", DINOV3_MODEL_IDS)
def test_dinov3_can_never_ship_weights(model_id):
    from visionservex.licensing import policy as P

    pol = P.get_policy(model_id)
    assert pol is not None
    assert pol.can_ship_weights is False


@pytest.mark.parametrize("model_id", DINOV3_MODEL_IDS)
def test_dinov3_requires_token_and_license(model_id):
    from visionservex.licensing import policy as P

    pol = P.get_policy(model_id)
    assert pol is not None
    assert pol.local_token_required is True
    assert pol.user_license_required is True
    assert pol.can_auto_download is False


@pytest.mark.parametrize("model_id", DINOV3_MODEL_IDS)
def test_dinov3_is_local(model_id):
    from visionservex.licensing import policy as P

    pol = P.get_policy(model_id)
    assert pol is not None
    assert pol.is_local is True


def test_dinov3_warning_text_byot():
    from visionservex.licensing import policy as P

    pol = P.get_policy("dinov3-vits16")
    assert pol is not None
    assert pol.warning_text.startswith("This model is gated")


def test_dinov3_not_apache_relabeled():
    from visionservex.licensing import policy as P

    for pol in P.iter_policies():
        if pol.model_id.startswith("dinov3"):
            assert "apache" not in pol.weights_license.lower(), (
                f"{pol.model_id} incorrectly labels weights as Apache"
            )


def test_dinov3_hf_repo_populated():
    from visionservex.licensing import policy as P

    for pol in P.iter_policies():
        if pol.model_id.startswith("dinov3"):
            assert pol.hf_repo and pol.hf_repo.startswith("facebook/")
