# SPDX-License-Identifier: Apache-2.0
"""v3.20/v3.21: foundation segmenters (SAM family) are never FULL-train-ready.

Full end-to-end SAM training is not wired (and would be illegitimate to claim).
v3.21 adds an HONEST, narrow capability: a frozen-encoder **mask-decoder**
fine-tune for HF SamModel models (``fine_tune_kind == 'frozen_encoder_decoder'``).
That is real (live-proven on sam-vit-base), so finetune-live is allowed ONLY when
it is exactly that decoder-only path — never a fake full-train finetune.

Weight-free.
"""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities

CAPS = {m: model_capabilities(m) for m in list_models()}
_FOUNDATION_SEG = {"sam", "sam2", "sam2.1", "mobilesam", "hq-sam", "efficientsam"}
FSEG = {m: c for m, c in CAPS.items() if c["family"] in _FOUNDATION_SEG}


def test_foundation_segmenters_exist():
    assert FSEG


def test_foundation_segmenters_are_never_full_train():
    for mid, c in FSEG.items():
        # full end-to-end training is not wired for SAM-style models
        assert not c["train_ready"], f"{mid} claims train_ready without a real trainer"
        assert not c["train_live_verified"], f"{mid} fake train-live"
        assert c["anastig_train_visibility"] != "show_train", mid


def test_any_sam_finetune_is_decoder_only_not_fake_full_train():
    for mid, c in FSEG.items():
        if c["fine_tune_ready"] or c["fine_tune_live_verified"]:
            # the ONLY honest SAM fine-tune is the frozen-encoder mask decoder
            assert c["fine_tune_kind"] == "frozen_encoder_decoder", (mid, c["fine_tune_kind"])
        if c["fine_tune_live_verified"]:
            # live decoder fine-tune is real -> show_finetune is honest here
            assert c["fine_tune_kind"] == "frozen_encoder_decoder", mid
        else:
            assert c["anastig_finetune_visibility"] != "show_finetune", mid
