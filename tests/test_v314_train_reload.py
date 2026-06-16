# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.14.0: detector training-RELIABILITY contract.

A detector family is only "train-ready" if a trained checkpoint can be reloaded
and used for inference. v3.14.0 fixes the LibreYOLO trained-checkpoint reload bug
(the inner module was left in training mode after the class-count rebuild, so the
head crashed at predict) and makes the capability table tell the truth:
``train_supported`` implies ``trained_checkpoint_predict_supported``.

These tests are weight-free: the load/reload paths are exercised with a fake
libreyolo class (no real weights, no GPU). The real end-to-end lifecycle is
proven live by ``tools/qa/v314_train_reload_matrix.py`` (artifact in
``docs/qa/v314_train_reload_matrix.json``) and ``tests/live/test_v314_train_reload_live.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_TRAINING_PATH_FILES = [
    "engines/libreyolo.py",
    "engines/rfdetr.py",
    "core/model.py",
    "data/yolo_dataset.py",
    "cli/training_commands.py",
]


class _Inner:
    """Stand-in for a libreyolo model's inner nn.Module (tracks training mode)."""

    def __init__(self) -> None:
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self


def _fake_libreyolo(captured: dict):
    """Return a fake libreyolo model class that records its construction args."""

    class _Fake:
        def __init__(self, model_path=None, size=None, device=None, **kw):
            captured["model_path"] = model_path
            captured["size"] = size
            captured["device"] = device
            # mimic libreyolo: inner module starts in TRAINING mode after a
            # class-count rebuild — the bug the v3.14 fix addresses.
            self.model = _Inner()

    return _Fake


def _patch_libreyolo(monkeypatch, captured: dict) -> None:
    import visionservex.engines.libreyolo as ly

    monkeypatch.setattr("visionservex.engines._stub.assert_modules", lambda *a, **k: None)
    monkeypatch.setattr(ly, "_load_libreyolo_class", lambda name: _fake_libreyolo(captured))


# ---------------------------------------------------------------------------
# 1. Capability truth: train_supported => trained_checkpoint_predict_supported
# ---------------------------------------------------------------------------


def test_train_capability_requires_reload_predict_support():
    from visionservex.core.model import _training_capabilities
    from visionservex.registry import default_registry

    offenders = []
    for e in default_registry().list():
        cap = _training_capabilities(e.id)
        if cap.get("train_supported") and not cap.get("trained_checkpoint_predict_supported"):
            offenders.append(e.id)
    assert not offenders, (
        "These report train_supported=True but NOT trained_checkpoint_predict_supported "
        f"(overclaim): {offenders}"
    )


# ---------------------------------------------------------------------------
# 2-5. Per-family lifecycle capability contracts
# ---------------------------------------------------------------------------


def _assert_full_lifecycle(model_id: str):
    from visionservex.core.model import _export_capabilities, _training_capabilities

    cap = _training_capabilities(model_id)
    assert cap["train_supported"] is True
    assert cap["finetune_supported"] is True
    assert cap["checkpoint_save_supported"] is True
    assert cap["checkpoint_load_supported"] is True
    assert cap["trained_checkpoint_predict_supported"] is True
    assert model_id in cap.get("validated_variants", [])
    assert _export_capabilities(model_id)["onnx"]["status"] == "supported"


def test_libreyolo_rtdetr_train_lifecycle_contract():
    _assert_full_lifecycle("libreyolo-rtdetr-r50")


def test_libreyolo_yolov9_train_lifecycle_contract():
    _assert_full_lifecycle("libreyolo-yolov9-s")


def test_libreyolo_yolox_not_marked_train_ready_if_reload_predict_fails():
    from visionservex.core.model import _training_capabilities

    cap = _training_capabilities("libreyolo-yolox-s")
    # The invariant: yolox is only marked train-ready BECAUSE reload+predict is
    # supported. v3.14.0 fixed the decode-after-reload bug, so this is now True
    # (it was the exact failure Anastig reported).
    if cap["train_supported"]:
        assert cap["trained_checkpoint_predict_supported"] is True, (
            "libreyolo-yolox-s must not report train_supported unless trained-checkpoint "
            "reload+predict works (it crashed pre-v3.14 with a decode/shape error)."
        )
    assert cap["trained_checkpoint_predict_supported"] is True


def test_libreyolo_dfine_train_lifecycle_contract_if_registered():
    from visionservex.registry import default_registry

    try:
        default_registry().get("libreyolo-dfine-n")
    except Exception:
        pytest.skip("libreyolo-dfine-n not registered")
    _assert_full_lifecycle("libreyolo-dfine-n")


# ---------------------------------------------------------------------------
# 6. Standalone HF D-FINE stays inference-only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mid", ["dfine-n", "dfine-s", "dfine-x-o365-coco"])
def test_standalone_dfine_hf_stays_inference_only(mid):
    from visionservex.core.model import _training_capabilities

    cap = _training_capabilities(mid)
    assert cap["train_supported"] is False
    assert cap["finetune_supported"] is False
    assert cap.get("trained_checkpoint_predict_supported") is False
    assert "HF_BACKEND" in cap["notes"]


# ---------------------------------------------------------------------------
# 7-9. Public reload API + no base-weight fallback (mocked load path)
# ---------------------------------------------------------------------------


def test_visionmodel_from_checkpoint_libreyolo(tmp_path, monkeypatch):
    from visionservex.core.model import VisionModel

    captured: dict = {}
    _patch_libreyolo(monkeypatch, captured)
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"\x00")

    m = VisionModel.from_checkpoint(str(ckpt), model_id="libreyolo-yolox-s", device="cpu")
    assert captured["model_path"] == str(ckpt)
    assert captured["size"] == "s"
    assert m._loaded is True
    # The v3.14 fix: the inner module is forced to eval after reload.
    assert m.engine._model.model.training is False


def test_visionmodel_load_checkpoint_libreyolo(tmp_path, monkeypatch):
    from visionservex.core.model import VisionModel

    captured: dict = {}
    _patch_libreyolo(monkeypatch, captured)
    ckpt = tmp_path / "last.pt"
    ckpt.write_bytes(b"\x00")

    m = VisionModel("libreyolo-rtdetr-r50")
    out = m.load_checkpoint(str(ckpt), device="cpu")
    assert out is m
    assert m._loaded is True
    assert captured["model_path"] == str(ckpt)
    assert m.engine._model.model.training is False


def test_no_base_weight_fallback_when_checkpoint_is_supplied(tmp_path, monkeypatch):
    import visionservex.runtime.downloads as dl
    from visionservex.engines import build_engine
    from visionservex.registry import default_registry

    captured: dict = {}
    _patch_libreyolo(monkeypatch, captured)

    called = {"dl": False}

    def _boom(*a, **k):
        called["dl"] = True
        raise AssertionError("base-weight download attempted while a checkpoint was supplied")

    monkeypatch.setattr(dl, "cached_path", _boom)
    monkeypatch.setattr(dl, "download", _boom)

    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"\x00")
    eng = build_engine(default_registry().get("libreyolo-yolox-s"))
    eng.load_checkpoint(str(ckpt), device="cpu")

    assert called["dl"] is False
    assert captured["model_path"] == str(ckpt)

    # And a missing checkpoint must raise cleanly (no base fallback).
    from visionservex.engines.base import MissingDependencyError

    eng2 = build_engine(default_registry().get("libreyolo-yolox-s"))
    with pytest.raises(MissingDependencyError):
        eng2.load_checkpoint(tmp_path / "nope.pt", device="cpu")
    assert called["dl"] is False


# ---------------------------------------------------------------------------
# 10. Export not overclaimed
# ---------------------------------------------------------------------------


def test_export_capability_not_overclaimed():
    from visionservex.core.model import _export_capabilities

    e = _export_capabilities("libreyolo-yolox-s")
    # ONNX is validated live (v3.14 QA matrix produces real .onnx files).
    assert e["onnx"]["status"] == "supported"
    # Formats we do not test must NOT be claimed 'supported'.
    for fmt in ("tensorrt", "openvino", "torchscript"):
        assert e.get(fmt, {}).get("status") != "supported", (
            f"{fmt} is claimed 'supported' for libreyolo without a tested path"
        )


# ---------------------------------------------------------------------------
# 11. Legal: no Ultralytics on the training/reload path
# ---------------------------------------------------------------------------


def test_no_ultralytics_imports():
    src = Path(__file__).resolve().parents[1] / "src" / "visionservex"
    forbidden = ("import ultralytics", "from ultralytics", "ultralytics.YOLO")
    for rel in _TRAINING_PATH_FILES:
        text = (src / rel).read_text()
        for pat in forbidden:
            assert pat not in text, f"{rel} contains forbidden {pat!r}"


# ---------------------------------------------------------------------------
# 12. YOLO-NAS never trainable
# ---------------------------------------------------------------------------


def test_yolonas_never_trainable():
    from visionservex.core.model import _training_capabilities
    from visionservex.engines.libreyolo import _TRAINABLE_FAMILIES

    assert "yolonas" not in _TRAINABLE_FAMILIES
    for mid in ["libreyolo-yolonas-s", "libreyolo-yolonas-m", "libreyolo-yolonas-l"]:
        cap = _training_capabilities(mid)
        assert cap["train_supported"] is False
        assert cap.get("trained_checkpoint_predict_supported") in (False, None)
