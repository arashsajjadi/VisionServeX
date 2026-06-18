# SPDX-License-Identifier: Apache-2.0
"""v3.19/v3.21: OpenMMLab families are never HOST-runnable; usable only via the
isolated Docker sidecar. Weight-free.

Host-native OpenMMLab is infeasible (py3.13/cu130), so no OpenMMLab model is ever
host live-verified or default-visible. v3.21 adds the sidecar path: a model proven
live through the OpenMMLab Docker sidecar (e.g. ``rtmpose-m``) is honestly visible
as ``show_*_sidecar`` with no blocker — but still NOT host live-verified.
"""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}
_OPENMMLAB_FAMILIES = {"internimage", "rtmdet", "rtmpose", "co-dino", "maskdino", "seem"}
OPENMMLAB = {m: c for m, c in CAPS.items() if c["family"] in _OPENMMLAB_FAMILIES}


def test_openmmlab_models_exist():
    assert OPENMMLAB


def test_openmmlab_is_never_host_live_verified():
    # The cardinal invariant: OpenMMLab cannot run in the host env. Sidecar-live
    # models prove inference through the sidecar, never the host path.
    for mid, c in OPENMMLAB.items():
        assert not c["live_verified_inference"], mid
        assert not c["live_verified_train"], mid


def test_non_sidecar_openmmlab_models_are_hidden_and_blocked():
    for mid, c in OPENMMLAB.items():
        if c["readiness_state"] in taxonomy.LIVE_SIDECAR_READY_STATES:
            continue  # sidecar-live: handled below
        assert c["anastig_visibility"] in ("hide", "blocked_admin_only"), (
            mid,
            c["anastig_visibility"],
        )
        assert c["blocker"], mid


def test_sidecar_live_openmmlab_is_visible_via_sidecar_only():
    sidecar = {
        m: c for m, c in OPENMMLAB.items()
        if c["readiness_state"] in taxonomy.LIVE_SIDECAR_READY_STATES
    }
    assert "rtmpose-m" in sidecar  # promoted in v3.21
    for mid, c in sidecar.items():
        assert c["anastig_visibility"].endswith("_sidecar"), (mid, c["anastig_visibility"])
        assert c["sidecar_required"] and c["sidecar_live"], mid
        assert not c["blocker"], mid
