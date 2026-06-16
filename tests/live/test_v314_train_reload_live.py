# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.14.0 LIVE lifecycle test (gated, CPU-safe).

Real train -> checkpoint -> reload (public API) -> predict -> export for each
permissive LibreYOLO detector. Skipped unless ``VSX_LIVE_TRAIN=1`` because it
performs real (tiny, 1-epoch) training and pulls base weights. Defaults to CPU
to avoid GPU/VRAM-saturation risk.

Run:
    VSX_LIVE_TRAIN=1 pytest tests/live/test_v314_train_reload_live.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

LIVE = os.environ.get("VSX_LIVE_TRAIN") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="set VSX_LIVE_TRAIN=1 to run live train/reload")

DEVICE = os.environ.get("VSX_LIVE_DEVICE", "cpu")
MODELS = [
    "libreyolo-yolox-s",
    "libreyolo-yolov9-s",
    "libreyolo-rtdetr-r50",
    "libreyolo-dfine-n",
]


def _make_dataset(root: Path, *, imgsz: int = 320, nc: int = 2) -> Path:
    from PIL import Image, ImageDraw

    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(6 if split == "train" else 3):
            cls = i % nc
            img = Image.new("RGB", (imgsz, imgsz), (30 + 20 * cls, 60, 90))
            ImageDraw.Draw(img).rectangle(
                [40 + 30 * cls, 50, 160 + 30 * cls, 140], fill=(200, 150, 40)
            )
            img.save(root / "images" / split / f"{split}_{i}.jpg", quality=90)
            (root / "labels" / split / f"{split}_{i}.txt").write_text(f"{cls} 0.5 0.5 0.35 0.28\n")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnc: {nc}\nnames: [a, b]\n"
    )
    return data_yaml


@pytest.mark.parametrize("model_id", MODELS)
def test_live_train_reload_predict(model_id, tmp_path):
    from PIL import Image

    from visionservex.core.model import VisionModel

    data_yaml = _make_dataset(tmp_path / "ds")
    test_img = Image.open(tmp_path / "ds" / "images" / "val" / "val_0.jpg").convert("RGB")

    res = VisionModel(model_id).train(
        str(data_yaml),
        epochs=1,
        device=DEVICE,
        imgsz=320,
        batch=2,
        project=str(tmp_path / "runs"),
        exist_ok=True,
    )
    assert res["status"] == "ok", res
    ckpt = res["best_checkpoint"] or res["last_checkpoint"]
    assert ckpt and Path(ckpt).is_file(), f"no checkpoint produced: {res}"

    # Public reload API -> predict must NOT crash (the v3.14 bug).
    trained = VisionModel.from_checkpoint(ckpt, model_id=model_id, device=DEVICE)
    pred = trained.predict(test_img, threshold=0.01)
    assert hasattr(pred, "detections"), pred  # valid (possibly empty) result

    # Export ONNX (advertised as supported).
    out = tmp_path / f"{model_id}.onnx"
    p = trained.export(format="onnx", output_path=str(out))
    assert Path(p).is_file()
    trained.unload()
