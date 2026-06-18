# SPDX-License-Identifier: Apache-2.0
"""v3.20: every model is either live or has an exact blocker — never vague. Weight-free."""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}


def test_each_model_is_live_or_has_a_specific_blocker():
    usable = taxonomy.LIVE_READY_STATES | taxonomy.LIVE_SIDECAR_READY_STATES
    for mid, c in CAPS.items():
        live = c["readiness_state"] in usable
        if live:
            assert not c["blocker"], f"{mid} is live but carries a blocker"
        else:
            assert c["blocker"] and len(c["blocker"]) > 10, (mid, c["blocker"])


def test_no_unknown_states():
    for mid, c in CAPS.items():
        assert c["readiness_state"] in taxonomy.READINESS_STATES, (mid, c["readiness_state"])
