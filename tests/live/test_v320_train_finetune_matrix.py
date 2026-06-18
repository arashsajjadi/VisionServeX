# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.20 LIVE: train/finetune matrix ↔ baked-evidence consistency (env-gated).

VSX_LIVE_TRAIN_FINETUNE_MATRIX=1 pytest tests/live/test_v320_train_finetune_matrix.py -q
"""

from __future__ import annotations

import os

import pytest

LIVE = os.environ.get("VSX_LIVE_TRAIN_FINETUNE_MATRIX") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="set VSX_LIVE_TRAIN_FINETUNE_MATRIX=1")


def test_baked_evidence_matches_committed_matrices():
    from visionservex.readiness import live_evidence as le

    assert set(le.LIVE_TRAIN_VERIFIED) == le.train_verified_from_matrix()
    assert set(le.LIVE_FINETUNE_VERIFIED) == le.finetune_verified_from_matrix()
    assert set(le.LIVE_RELOAD_VERIFIED) == le.reload_verified_from_matrix()
    assert set(le.LIVE_EXPORT_VERIFIED) == le.export_verified_from_matrix()
    assert len(le.LIVE_FINETUNE_VERIFIED) >= 10  # all embed models head-finetuned
