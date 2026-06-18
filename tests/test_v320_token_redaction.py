# SPDX-License-Identifier: Apache-2.0
"""v3.20: tokens are never printed/logged/committed. Weight-free."""

from __future__ import annotations

import pytest

from visionservex.core.model import list_models, model_capabilities

CAPS = {m: model_capabilities(m) for m in list_models()}


def test_redaction_hides_the_middle():
    from visionservex.hf_auth import hf_redact_token

    sample = "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    red = hf_redact_token(sample)
    assert red != sample and sample[3:-2] not in red


def test_no_capability_payload_contains_a_raw_token():
    from visionservex.hf_auth import hf_get_token

    raw = hf_get_token(redact=False)
    if not raw:
        pytest.skip("no local HF token configured")
    for mid, c in CAPS.items():
        assert raw not in repr(c), f"{mid} leaked a raw token"


def test_token_never_logged_flag_is_true():
    for mid, c in CAPS.items():
        assert c["token_never_logged"] is True, mid


def test_no_token_shaped_strings_in_capabilities():
    for mid, c in CAPS.items():
        for k, v in c.items():
            if isinstance(v, str) and v.startswith("hf_") and len(v) > 20:
                pytest.fail(f"{mid}.{k} looks like a raw token")
