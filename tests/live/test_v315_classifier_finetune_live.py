# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.15.0 LIVE: torchvision classifier full lifecycle (gated, CPU-safe).

pretrained inference -> ImageFolder fine-tune -> checkpoint -> reload (public
API) -> predict -> ONNX export. Skipped unless ``VSX_LIVE_TRAIN=1`` (downloads
ImageNet weights, runs real training). Defaults to CPU (no VRAM risk).

    VSX_LIVE_TRAIN=1 pytest tests/live/test_v315_classifier_finetune_live.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

LIVE = os.environ.get("VSX_LIVE_TRAIN") == "1"
pytestmark = pytest.mark.skipif(
    not LIVE, reason="set VSX_LIVE_TRAIN=1 to run live classifier lifecycle"
)

DEVICE = os.environ.get("VSX_LIVE_DEVICE", "cpu")


def _imagefolder(root: Path) -> Path:
    from PIL import Image

    for cls, col in (("red", (200, 40, 40)), ("blue", (40, 40, 200))):
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            Image.new("RGB", (224, 224), col).save(d / f"{i}.jpg")
    return root


@pytest.mark.parametrize("model_id", ["torchvision-resnet18", "torchvision-mobilenet-v2"])
def test_live_classifier_lifecycle(model_id, tmp_path):
    from PIL import Image

    from visionservex.core.model import VisionModel

    # 1. pretrained inference
    m = VisionModel(model_id)
    pred = m.predict(Image.new("RGB", (256, 256), (200, 40, 40)), top_k=3)
    assert pred.kind == "classification" and len(pred.top_k) == 3
    m.unload()

    # 2. fine-tune
    ds = _imagefolder(tmp_path / "ds")
    res = VisionModel(model_id).train(
        str(ds), epochs=1, batch=4, device=DEVICE, project=str(tmp_path / "runs")
    )
    assert res["status"] == "ok"
    ckpt = res["best_checkpoint"]
    assert Path(ckpt).is_file()
    assert res["artifacts"]["classes"] == ["blue", "red"]

    # 3. reload (public API) + predict the fine-tuned classes
    trained = VisionModel.from_checkpoint(ckpt, model_id=model_id, device=DEVICE)
    out = trained.predict(Image.new("RGB", (224, 224), (200, 40, 40)), top_k=2)
    labels = {label for label, _ in out.top_k}
    assert labels <= {"red", "blue"}, (
        f"fine-tuned model should predict dataset classes, got {labels}"
    )

    # 4. export ONNX
    onnx = tmp_path / f"{model_id}.onnx"
    p = trained.export(format="onnx", output_path=str(onnx))
    assert Path(p).is_file()
    trained.unload()
