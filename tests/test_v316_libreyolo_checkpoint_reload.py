# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.16.0: LibreYOLO checkpoint contract — best.pt fallback + trained-imgsz reload.

Weight-free: the libreyolo class + module gate are mocked so no real weights or
GPU are needed. Proves the two v3.16 reliability fixes at the unit level.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_best_checkpoint_falls_back_to_last(tmp_path):
    """best.pt is only written when val mAP improves; the result's best_checkpoint
    must fall back to last.pt (which always exists) so reload never points at a
    missing file."""
    from visionservex.engines.libreyolo import _normalize_train_result

    wdir = tmp_path / "weights"
    wdir.mkdir()
    (wdir / "last.pt").write_bytes(b"\x00")  # only last.pt exists (no best.pt)
    raw = {
        "save_dir": str(tmp_path),
        "best_checkpoint": str(wdir / "best.pt"),  # does NOT exist
        "last_checkpoint": str(wdir / "last.pt"),
    }
    res = _normalize_train_result(
        raw, model_id="libreyolo-yolox-s", variant="yolox-s", data_yaml=Path("d.yaml"), hours=0.1
    )
    assert res["best_checkpoint"] == str(wdir / "last.pt")  # fell back
    assert res["checkpoint"] == str(wdir / "last.pt")
    assert Path(res["checkpoint"]).is_file()


def test_trained_imgsz_read_and_applied(tmp_path, monkeypatch):
    """load_checkpoint reads the training imgsz from the checkpoint config and sets
    the model's input_size, so predict() infers at the trained resolution."""
    import torch  # available with the libreyolo extra

    import visionservex.engines.libreyolo as ly
    from visionservex.engines import build_engine
    from visionservex.registry import default_registry

    captured: dict = {}

    class _Inner:
        def __init__(self):
            self.training = True

        def eval(self):
            self.training = False

    class _Fake:
        def __init__(self, model_path=None, size=None, device=None, **kw):
            captured["model_path"] = model_path
            self.input_size = 640  # native default
            self.model = _Inner()

    monkeypatch.setattr("visionservex.engines._stub.assert_modules", lambda *a, **k: None)
    monkeypatch.setattr(ly, "_load_libreyolo_class", lambda name: _Fake)

    ckpt = tmp_path / "best.pt"
    torch.save({"config": {"imgsz": 320}, "model": {}, "nc": 2}, str(ckpt))

    eng = build_engine(default_registry().get("libreyolo-yolox-s"))
    eng.load_checkpoint(str(ckpt), device="cpu")

    assert eng._trained_imgsz == 320
    assert eng._model.input_size == 320  # predict will infer at the trained imgsz
    assert captured["model_path"] == str(ckpt)
    assert eng._model.model.training is False  # eval forced


def test_missing_checkpoint_clean_error_no_base_fallback(tmp_path, monkeypatch):
    import visionservex.runtime.downloads as dl
    from visionservex.engines import build_engine
    from visionservex.engines.base import MissingDependencyError
    from visionservex.registry import default_registry

    called = {"dl": False}

    def _boom(*a, **k):
        called["dl"] = True
        raise AssertionError("base-weight download attempted with checkpoint supplied")

    monkeypatch.setattr(dl, "cached_path", _boom)
    monkeypatch.setattr(dl, "download", _boom)

    eng = build_engine(default_registry().get("libreyolo-yolox-s"))
    with pytest.raises(MissingDependencyError):
        eng.load_checkpoint(tmp_path / "nope.pt", device="cpu")
    assert called["dl"] is False
