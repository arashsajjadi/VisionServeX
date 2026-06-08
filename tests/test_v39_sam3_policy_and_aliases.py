# SPDX-License-Identifier: Apache-2.0
"""v3.9 — SAM3/SAM3.1 policy: correct bucket, BYOT flags, no-ship invariant."""

from __future__ import annotations

import pytest

SAM3_MODEL_IDS = [
    "sam3-base",
    "sam3-image",
    "sam3-text-prompt",
    "sam3.1-base",
    "sam3.1-image",
    "sam3.1-text-prompt",
    "sam3.1-video",
]


@pytest.mark.parametrize("model_id", SAM3_MODEL_IDS)
def test_sam3_policy_bucket(model_id):
    from visionservex.licensing import policy as P

    pol = P.get_policy(model_id)
    assert pol is not None, f"No policy for {model_id}"
    assert pol.final_policy == "byot_license_required"


@pytest.mark.parametrize("model_id", SAM3_MODEL_IDS)
def test_sam3_can_never_ship_weights(model_id):
    from visionservex.licensing import policy as P

    pol = P.get_policy(model_id)
    assert pol is not None
    assert pol.can_ship_weights is False


@pytest.mark.parametrize("model_id", SAM3_MODEL_IDS)
def test_sam3_requires_token_and_license(model_id):
    from visionservex.licensing import policy as P

    pol = P.get_policy(model_id)
    assert pol is not None
    assert pol.local_token_required is True
    assert pol.user_license_required is True
    assert pol.can_auto_download is False


def test_sam3_not_apache_relabeled():
    from visionservex.licensing import policy as P

    for pol in P.iter_policies():
        if pol.model_id.startswith("sam3"):
            assert "apache" not in pol.weights_license.lower(), (
                f"{pol.model_id} incorrectly labeled as Apache"
            )


def test_sam3_hf_repo_is_facebook():
    from visionservex.licensing import policy as P

    for pol in P.iter_policies():
        if pol.model_id.startswith("sam3"):
            assert pol.hf_repo.startswith("facebook/")


def test_sam3_datasets_not_in_policy_as_models():
    """SACo-Gold / SA-FARI are not runtime models; they must not appear as core model rows."""
    from visionservex.licensing import policy as P

    model_ids = {pol.model_id for pol in P.iter_policies()}
    dataset_ids = {"saco-gold", "saco-silver", "saco-veval", "sa-fari"}
    overlap = model_ids & dataset_ids
    assert not overlap, f"Dataset IDs wrongly appear as model rows: {overlap}"
