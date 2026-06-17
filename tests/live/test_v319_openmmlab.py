# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.19 LIVE: OpenMMLab sidecar probe (env-gated). Host-native is infeasible.

    VSX_LIVE_OPENMMLAB=1 pytest tests/live/test_v319_openmmlab.py -q

Without a running Docker sidecar these models stay blocked; this test documents
that contract and only probes the sidecar when explicitly enabled.
"""

from __future__ import annotations

import os

import pytest

LIVE = os.environ.get("VSX_LIVE_OPENMMLAB") == "1"
pytestmark = [
    pytest.mark.skipif(not LIVE, reason="set VSX_LIVE_OPENMMLAB=1"),
    pytest.mark.real_model,
]


def test_openmmlab_requires_sidecar_url():
    # Host-native mmcv is infeasible (py3.13/cu130); the only path is the sidecar.
    url = os.environ.get("VISIONSERVEX_OPENMMLAB_SIDECAR_URL")
    if not url:
        pytest.skip("no OpenMMLab sidecar URL set; run `visionservex openmmlab docker-run`")
    import urllib.request

    with urllib.request.urlopen(f"{url}/health", timeout=5) as resp:
        assert resp.status == 200
