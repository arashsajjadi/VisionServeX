# SPDX-License-Identifier: Apache-2.0
"""v3.19: every model is either live-proven or carries an exact blocker. Weight-free.

This is the core honesty invariant of the operationalization sprint: there is no
silent middle ground — a model is live-verified, OR it has a precise blocker that
says exactly why it is hidden.
"""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}


def test_every_model_is_live_or_blocked_never_neither():
    for mid, c in CAPS.items():
        is_live = c["readiness_state"] in taxonomy.LIVE_READY_STATES
        has_blocker = bool(c["blocker"])
        assert is_live != has_blocker or (is_live and not has_blocker), mid
        # precise: live -> no blocker; not live -> a blocker string exists
        if is_live:
            assert not has_blocker, f"{mid} is live but still carries a blocker"
        else:
            assert has_blocker, f"{mid} is not live and has NO blocker (vague state)"


def test_blocker_is_nonempty_and_specific_for_blocked_models():
    for mid, c in CAPS.items():
        if c["readiness_state"] not in taxonomy.LIVE_READY_STATES:
            assert c["blocker"] and len(c["blocker"]) > 10, (mid, c["blocker"])


def test_default_visible_models_are_all_live():
    for mid, c in CAPS.items():
        if c["anastig_visibility"] in (
            "show_train",
            "show_inference",
            "show_embedding",
            "show_segmentation",
        ):
            assert c["live_verified_inference"] or c["live_verified_train"], mid
