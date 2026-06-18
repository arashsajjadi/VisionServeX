# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.19 LIVE: DEIM/DEIMv2/RT-DETRv4 custom-loader probe (env-gated).

    VSX_LIVE_CUSTOM_LOADERS=1 pytest tests/live/test_v319_custom_loaders.py -q

In-process loading is infeasible (non-HF configs / torch-pin conflict); these
stay CUSTOM_LOADER_REQUIRED. This test asserts the honest blocked contract.
"""

from __future__ import annotations

import os

import pytest

LIVE = os.environ.get("VSX_LIVE_CUSTOM_LOADERS") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="set VSX_LIVE_CUSTOM_LOADERS=1")


def test_deim_rtdetrv4_are_not_in_process_loadable():
    from visionservex.core.model import list_models, model_capabilities
    from visionservex.readiness import taxonomy

    for m in list_models():
        c = model_capabilities(m)
        if c["family"] in ("deim", "rtdetrv4"):
            assert c["readiness_state"] in (
                taxonomy.CUSTOM_LOADER_REQUIRED,
                taxonomy.CATALOG_ONLY_ENGINE_NOT_WIRED,
            )
            assert not c["live_verified_inference"]
