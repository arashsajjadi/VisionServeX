# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.16.0: legal firewall — no Ultralytics/AGPL/GPL in runtime; YOLO-NAS blocked."""

from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src" / "visionservex"
_RUNTIME_DIRS = [_SRC / "engines", _SRC / "core", _SRC / "data", _SRC / "runtime"]


def test_no_ultralytics_in_runtime():
    forbidden = ("import ultralytics", "from ultralytics", "ultralytics.YOLO")
    for d in _RUNTIME_DIRS:
        for f in d.rglob("*.py"):
            text = f.read_text()
            for pat in forbidden:
                assert pat not in text, f"{f} imports Ultralytics ({pat!r})"


def test_postprocess_nms_is_pure_numpy_no_copyleft():
    """The NMS safety net must not pull a copyleft/torch/ultralytics dependency."""
    text = (_SRC / "runtime" / "postprocess.py").read_text()
    assert "import ultralytics" not in text
    assert "import torch" not in text  # pure numpy NMS (no torch/torchvision import)
    assert "import torchvision" not in text
    from visionservex.runtime.postprocess import class_aware_nms  # importable

    assert callable(class_aware_nms)


def test_yolonas_never_trainable():
    from visionservex.core.model import _training_capabilities
    from visionservex.engines.libreyolo import _TRAINABLE_FAMILIES

    assert "yolonas" not in _TRAINABLE_FAMILIES
    for mid in ("libreyolo-yolonas-s", "libreyolo-yolonas-m", "libreyolo-yolonas-l"):
        cap = _training_capabilities(mid)
        assert cap["train_supported"] is False
        assert cap.get("trained_checkpoint_predict_supported") in (False, None)
