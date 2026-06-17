# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.19 LIVE: train-matrix ↔ baked-evidence consistency. Env-gated.

VSX_LIVE_TRAIN_MATRIX=1 pytest tests/live/test_v319_live_train_matrix.py -q
"""

from __future__ import annotations

import os

import pytest

LIVE = os.environ.get("VSX_LIVE_TRAIN_MATRIX") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="set VSX_LIVE_TRAIN_MATRIX=1")


def test_baked_train_evidence_matches_committed_matrices():
    from visionservex.readiness import live_evidence

    assert set(live_evidence.LIVE_TRAIN_VERIFIED) == live_evidence.train_verified_from_matrix()
    # the 8 RF-DETR variants are part of the train-verified set
    rfdetr = {m for m in live_evidence.LIVE_TRAIN_VERIFIED if m.startswith("rfdetr")}
    assert len(rfdetr) == 8, rfdetr
