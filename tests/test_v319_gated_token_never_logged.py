# SPDX-License-Identifier: Apache-2.0
"""v3.19: gated models require a token and never leak it. Weight-free."""

from __future__ import annotations

import pytest

from visionservex.core.model import list_models, model_capabilities

CAPS = {m: model_capabilities(m) for m in list_models()}
GATED = {m: c for m, c in CAPS.items() if c["gated"]}


def test_gated_models_exist_and_require_token():
    assert GATED
    for mid, c in GATED.items():
        assert c["requires_token"] is True, mid
        assert not c["commercial_safe"], mid


def test_redaction_actually_redacts():
    from visionservex.hf_auth import hf_redact_token

    sample = "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    red = hf_redact_token(sample)
    assert red != sample
    assert sample[3:-2] not in red  # the middle is hidden


def test_no_raw_token_in_any_capability_payload():
    from visionservex.hf_auth import hf_get_token

    raw = hf_get_token(redact=False)
    if not raw:
        pytest.skip("no local HF token configured")
    for mid, c in CAPS.items():
        assert raw not in repr(c), f"{mid} capability payload leaked a raw token"


def test_capabilities_never_call_for_raw_token():
    # model_capabilities must not surface any token-bearing field at all.
    for mid, c in CAPS.items():
        for k, v in c.items():
            if isinstance(v, str) and v.startswith("hf_") and len(v) > 20:
                pytest.fail(f"{mid}.{k} looks like a raw token: {v[:6]}...")
