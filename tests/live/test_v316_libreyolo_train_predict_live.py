# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.16.0 LIVE: LibreYOLO train→reload→predict-confident-boxes→NMS (gated, CPU).

Proves the eval/=predict fix end-to-end for the validated variants: after adequate
training, predict() at the DEFAULT threshold returns confident, post-NMS boxes.
Skipped unless ``VSX_LIVE_LIBREYOLO_TRAIN=1`` (real training + base-weight pull).

    VSX_LIVE_LIBREYOLO_TRAIN=1 pytest tests/live/test_v316_libreyolo_train_predict_live.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

LIVE = os.environ.get("VSX_LIVE_LIBREYOLO_TRAIN") == "1"
pytestmark = pytest.mark.skipif(
    not LIVE, reason="set VSX_LIVE_LIBREYOLO_TRAIN=1 to run live LibreYOLO lifecycle"
)

DEVICE = os.environ.get("VSX_LIVE_DEVICE", "cpu")
EPOCHS = int(os.environ.get("VSX_LIVE_EPOCHS", "25"))


def _dataset(root: Path, *, nc: int = 2, imgsz: int = 320) -> Path:
    from PIL import Image, ImageDraw

    colors = [(210, 60, 60), (60, 60, 210)]
    for split, n in (("train", 24), ("val", 8)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cls = i % nc
            img = Image.new("RGB", (imgsz, imgsz), (30, 40, 50))
            x0, y0 = 40 + (i * 7) % 120, 40 + (i * 5) % 120
            ImageDraw.Draw(img).rectangle([x0, y0, x0 + 90, y0 + 80], fill=colors[cls])
            img.save(root / "images" / split / f"{split}_{i}.jpg", quality=92)
            cx, cy = (x0 + 45) / imgsz, (y0 + 40) / imgsz
            (root / "labels" / split / f"{split}_{i}.txt").write_text(
                f"{cls} {cx:.5f} {cy:.5f} {90 / imgsz:.5f} {80 / imgsz:.5f}\n"
            )
    yaml = root / "data.yaml"
    yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnc: {nc}\nnames: [a, b]\n"
    )
    return yaml


@pytest.mark.parametrize(
    "model_id", ["libreyolo-yolox-s", "libreyolo-yolov9-s", "libreyolo-rtdetr-r50"]
)
def test_live_train_reload_predict_confident(model_id, tmp_path):
    from PIL import Image

    from visionservex.core.model import VisionModel

    yaml = _dataset(tmp_path / "ds")
    val_img = Image.open(tmp_path / "ds" / "images" / "val" / "val_0.jpg").convert("RGB")

    res = VisionModel(model_id).train(
        str(yaml),
        epochs=EPOCHS,
        batch=4,
        imgsz=320,
        device=DEVICE,
        project=str(tmp_path / "runs"),
        exist_ok=True,
    )
    assert res["status"] == "ok"
    ckpt = res["checkpoint"] or res["best_checkpoint"]
    assert ckpt and Path(ckpt).is_file()

    trained = VisionModel.from_checkpoint(ckpt, model_id=model_id, device=DEVICE)
    # the eval/=predict fix: confident boxes at the DEFAULT threshold (0.25)
    pred = trained.predict(val_img)
    assert len(pred.detections) >= 1, f"{model_id}: predict returned 0 boxes at default threshold"
    assert pred.metadata["post_nms_count"] <= pred.metadata["raw_count"]  # NMS applied

    # raw access for debugging preserves proposals
    raw = trained.predict(val_img, threshold=0.001, return_raw=True)
    nms = trained.predict(val_img, threshold=0.001)
    assert nms.metadata["post_nms_count"] <= raw.metadata["raw_count"]
    trained.unload()
