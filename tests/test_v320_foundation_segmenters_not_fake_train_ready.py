# SPDX-License-Identifier: Apache-2.0
"""v3.20: foundation segmenters (SAM family) are inference-only — never fake train-ready.

Weight-free.
"""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities

CAPS = {m: model_capabilities(m) for m in list_models()}
_FOUNDATION_SEG = {"sam", "sam2", "sam2.1", "mobilesam", "hq-sam", "efficientsam"}
FSEG = {m: c for m, c in CAPS.items() if c["family"] in _FOUNDATION_SEG}


def test_foundation_segmenters_exist():
    assert FSEG


def test_foundation_segmenters_are_not_train_or_finetune_live():
    for mid, c in FSEG.items():
        assert not c["train_live_verified"], f"{mid} fake train-live"
        assert not c["fine_tune_live_verified"], f"{mid} fake finetune-live"
        assert c["anastig_train_visibility"] != "show_train", mid
        assert c["anastig_finetune_visibility"] != "show_finetune", mid


def test_foundation_segmenters_are_not_train_ready():
    for mid, c in FSEG.items():
        # full end-to-end training is not wired for SAM-style models
        assert not c["train_ready"], f"{mid} claims train_ready without a real trainer"
