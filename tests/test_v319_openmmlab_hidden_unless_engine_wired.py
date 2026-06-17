# SPDX-License-Identifier: Apache-2.0
"""v3.19: OpenMMLab families stay hidden until the engine is really wired. Weight-free.

Host-native OpenMMLab is infeasible (py3.13/cu130); these models must never appear
default-visible to Anastig users.
"""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities

CAPS = {m: model_capabilities(m) for m in list_models()}
_OPENMMLAB_FAMILIES = {"internimage", "rtmdet", "rtmpose", "co-dino", "maskdino", "seem"}
OPENMMLAB = {m: c for m, c in CAPS.items() if c["family"] in _OPENMMLAB_FAMILIES}


def test_openmmlab_models_exist():
    assert OPENMMLAB


def test_openmmlab_models_are_not_default_visible():
    for mid, c in OPENMMLAB.items():
        assert c["anastig_visibility"] in ("hide", "blocked_admin_only"), (
            mid,
            c["anastig_visibility"],
        )
        assert not c["live_verified_inference"], mid
        assert not c["live_verified_train"], mid


def test_openmmlab_models_carry_a_blocker():
    for mid, c in OPENMMLAB.items():
        assert c["blocker"], mid
