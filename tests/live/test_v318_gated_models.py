# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.18 LIVE: gated (BYOT) models — token required, never leaked.

    VSX_LIVE_GATED_MODELS=1 pytest tests/live/test_v318_gated_models.py -q

The capability-level guarantees (requires_token, GATED state, no raw token in the
capability dict) are enforced unconditionally; the network access probe is gated.
"""

from __future__ import annotations

import os

import pytest

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import taxonomy

LIVE = os.environ.get("VSX_LIVE_GATED_MODELS") == "1"

GATED = {m: model_capabilities(m) for m in list_models() if model_capabilities(m)["gated"]}


def test_gated_models_exist_and_require_token():
    assert GATED
    for mid, cap in GATED.items():
        assert cap["requires_token"] is True, mid
        assert not cap["commercial_safe"], mid


def test_gated_byot_state_and_visibility():
    for mid, cap in GATED.items():
        if cap["readiness_state"] == taxonomy.GATED_TOKEN_REQUIRED:
            assert cap["anastig_visibility"] == "show_token_required", mid


def test_capability_dict_never_contains_a_raw_token():
    # A model_capabilities() payload must never embed a raw HF token.
    from visionservex.hf_auth import hf_get_token

    raw = hf_get_token(redact=False)
    if not raw:
        pytest.skip("no local HF token configured")
    for mid, cap in GATED.items():
        blob = repr(cap)
        assert raw not in blob, f"{mid} capability payload leaked a raw token"


@pytest.mark.skipif(not LIVE, reason="set VSX_LIVE_GATED_MODELS=1")
@pytest.mark.real_model
def test_gated_access_probe_is_redacted():
    from visionservex.hf_auth import hf_get_token, hf_model_access_status, hf_redact_token

    for mid in GATED:
        status = hf_model_access_status(mid)
        assert "state" in status
        raw = hf_get_token(redact=False)
        if raw:
            assert raw not in repr(status), f"{mid} access status leaked a raw token"
            assert hf_redact_token(raw) != raw  # redaction actually changes it
