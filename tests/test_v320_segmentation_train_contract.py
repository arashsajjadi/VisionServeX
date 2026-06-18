# SPDX-License-Identifier: Apache-2.0
"""v3.20: segmentation train contract. Weight-free.

Only architectures with a real segmentation trainer (RF-DETR-Seg) are
segmentation-train-ready-live; foundation segmenters are inference-only.
"""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities

CAPS = {m: model_capabilities(m) for m in list_models()}
SEG = {
    m: c
    for m, c in CAPS.items()
    if c["task"] in ("segment", "foundation_segment", "grounded_segment")
}


def test_segmentation_train_live_models_are_rfdetr_seg():
    train_live = {m for m, c in SEG.items() if c["train_live_verified"]}
    assert train_live, "expected RF-DETR-Seg variants to be segmentation-train-live"
    for m in train_live:
        assert CAPS[m]["family"] == "rfdetr", m
        assert CAPS[m]["reload_live_verified"], m
        assert CAPS[m]["export_live_verified"], m


def test_foundation_segmenters_not_train_live():
    fseg = {"sam", "sam2", "sam2.1", "mobilesam", "hq-sam", "efficientsam"}
    for m, c in SEG.items():
        if c["family"] in fseg:
            assert not c["train_live_verified"], m


def test_live_segmentation_models_have_valid_state():
    from visionservex.readiness import taxonomy

    for m, c in SEG.items():
        if c["inference_live_verified"]:
            assert c["readiness_state"] in (
                taxonomy.SEGMENTATION_READY_LIVE,
                taxonomy.TRAIN_READY_LIVE,
            ), (m, c["readiness_state"])
