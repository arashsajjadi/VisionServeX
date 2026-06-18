# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.20 LIVE: gated-model BYOT contract + token never leaked (env-gated network probe).

VSX_LIVE_GATED=1 pytest tests/live/test_v320_gated_models.py -q
"""

from __future__ import annotations

import os

import pytest

from visionservex.core.model import list_models, model_capabilities

GATED = [m for m in list_models() if model_capabilities(m)["gated"]]


def test_gated_contract_unconditional():
    assert GATED
    for m in GATED:
        c = model_capabilities(m)
        assert c["requires_token"] and not c["commercial_safe"]
        assert c["anastig_train_visibility"] != "show_train"


@pytest.mark.skipif(os.environ.get("VSX_LIVE_GATED") != "1", reason="set VSX_LIVE_GATED=1")
@pytest.mark.real_model
def test_access_probe_is_redacted():
    from visionservex.hf_auth import hf_get_token, hf_model_access_status, hf_redact_token

    raw = hf_get_token(redact=False)
    for m in GATED:
        status = hf_model_access_status(m)
        assert "state" in status
        if raw:
            assert raw not in repr(status)
            assert hf_redact_token(raw) != raw
