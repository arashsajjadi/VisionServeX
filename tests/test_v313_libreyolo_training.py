# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.13.0: LibreYOLO detector training support.

These tests prove the *training contract* — capability flags, the engine's
``train``/``load_checkpoint``/``export`` surface, the normalized result shape,
the YOLO-NAS legal exclusion, the standalone-HF-D-FINE inference-only guarantee,
and the no-Ultralytics rule. They are intentionally weight-free and (where the
load path is exercised) libreyolo-import-free via mocking, so they run in CI
without GPUs or model downloads. Real end-to-end training is a separate live
smoke gated behind ``VSX_LIVE_LIBREYOLO_TRAIN=1`` (see docs/libreyolo_training.md).
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

LIBREYOLO_TRAIN_IDS = ["libreyolo-yolox-s", "libreyolo-yolov9-s", "libreyolo-rtdetr-r50"]
# Source files that make up the LibreYOLO training path. These — and only these
# — must never import Ultralytics (the pre-existing optional ultralytics
# *benchmark-comparison* path in cli/benchmark_commands.py is out of scope and
# is not a runtime dependency).
_TRAINING_PATH_FILES = [
    "engines/libreyolo.py",
    "core/model.py",
    "data/yolo_dataset.py",
    "cli/training_commands.py",
]


def _make_yolo_dataset(root: Path) -> Path:
    """Create a minimal, valid YOLO dataset directory (dirs + data.yaml)."""
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "data.yaml").write_text("train: images/train\nval: images/val\nnc: 1\nnames: [obj]\n")
    return root


# ---------------------------------------------------------------------------
# 1. Capability flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mid", LIBREYOLO_TRAIN_IDS)
def test_libreyolo_train_capability_is_true(mid):
    from visionservex.core.model import _training_capabilities

    cap = _training_capabilities(mid)
    assert cap["train_supported"] is True
    assert cap["finetune_supported"] is True
    assert cap["resume_supported"] is True
    assert cap["checkpoint_save_supported"] is True
    assert cap["checkpoint_load_supported"] is True
    assert cap["supported_dataset_formats"] == ["yolo"]
    assert cap["required_extra"] == "libreyolo"


def test_libreyolo_export_capability():
    from visionservex.core.model import _export_capabilities

    info = _export_capabilities("libreyolo-yolox-s")
    assert info["onnx"]["status"] == "supported"


# ---------------------------------------------------------------------------
# 2. Engine surface
# ---------------------------------------------------------------------------


def test_libreyolo_engine_has_train_method():
    from visionservex.engines.libreyolo import LibreYOLOEngine

    assert callable(getattr(LibreYOLOEngine, "train", None))
    assert callable(getattr(LibreYOLOEngine, "load_checkpoint", None))
    assert callable(getattr(LibreYOLOEngine, "export", None))


def test_visionmodel_has_train_method():
    from visionservex.core.model import VisionModel

    assert callable(getattr(VisionModel, "train", None))


# ---------------------------------------------------------------------------
# 3. YOLO-NAS / non-commercial exclusion (legal hard rule)
# ---------------------------------------------------------------------------


def test_yolonas_excluded_from_trainable_families():
    from visionservex.engines.libreyolo import _TRAINABLE_FAMILIES

    assert "yolonas" not in _TRAINABLE_FAMILIES
    assert set(_TRAINABLE_FAMILIES) == {"yolox", "yolov9", "rtdetr", "dfine"}


def test_yolonas_training_capability_is_false():
    from visionservex.core.model import _training_capabilities

    cap = _training_capabilities("libreyolo-yolonas-s")
    assert cap["train_supported"] is False
    assert cap["finetune_supported"] is False


def test_libreyolo_engine_rejects_yolonas_training(tmp_path):
    from visionservex.engines import build_engine
    from visionservex.engines.libreyolo import TrainingNotSupportedError
    from visionservex.registry import default_registry

    eng = build_engine(default_registry().get("libreyolo-yolox-s"))
    # Swap the entry id to a non-commercial family; train() must reject it
    # BEFORE touching the dataset or loading any weights.
    eng.entry = types.SimpleNamespace(id="libreyolo-yolonas-s")
    with pytest.raises(TrainingNotSupportedError):
        eng.train(tmp_path / "data.yaml")


# ---------------------------------------------------------------------------
# 4. Checkpoint reload contract
# ---------------------------------------------------------------------------


def test_libreyolo_checkpoint_reload_contract(tmp_path, monkeypatch):
    import visionservex.engines.libreyolo as ly
    from visionservex.engines import build_engine
    from visionservex.engines.base import MissingDependencyError
    from visionservex.registry import default_registry

    captured: dict = {}

    class _FakeModel:
        def __init__(self, model_path=None, size=None, device=None, **kw):
            captured["model_path"] = model_path
            captured["size"] = size
            captured["device"] = device

    # Return our fake instead of the real libreyolo class, and skip the
    # libreyolo import gate so the test needs no libreyolo install.
    monkeypatch.setattr(ly, "_load_libreyolo_class", lambda name: _FakeModel)
    eng = build_engine(default_registry().get("libreyolo-yolox-s"))
    eng.real_modules = ()

    # Missing checkpoint -> clean error, and the model is never constructed.
    with pytest.raises(MissingDependencyError):
        eng.load_checkpoint(tmp_path / "nope.pt")
    assert captured == {}

    # Real checkpoint -> passed straight into the libreyolo constructor; the
    # family/size come from the model id, NOT from the file. No base-weight
    # fallback (the override is the sole weight source).
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"\x00")
    eng.load_checkpoint(ckpt, device="cpu")
    assert captured["model_path"] == str(ckpt)
    assert captured["size"] == "s"
    assert eng._checkpoint_override == ckpt
    assert eng._real_ready is True


# ---------------------------------------------------------------------------
# 5. Train result contract (mock the underlying libreyolo trainer)
# ---------------------------------------------------------------------------


def test_libreyolo_train_result_contract(tmp_path):
    from visionservex.engines import build_engine
    from visionservex.registry import default_registry

    ds = _make_yolo_dataset(tmp_path)
    save_dir = tmp_path / "runs" / "exp"
    fake_raw = {
        "save_dir": str(save_dir),
        "best_checkpoint": str(save_dir / "weights" / "best.pt"),
        "last_checkpoint": str(save_dir / "weights" / "last.pt"),
        "best_mAP50": 0.51,
        "best_mAP50_95": 0.33,
        "best_epoch": 7,
        "final_loss": 1.23,
        "epoch_losses": [3.0, 2.0, 1.23],
    }
    captured: dict = {}

    def _fake_train(data, **kw):
        captured["data"] = data
        captured["kw"] = kw
        return fake_raw

    eng = build_engine(default_registry().get("libreyolo-yolox-s"))
    # Pre-load a fake model so train() skips the real load path entirely.
    eng._model = types.SimpleNamespace(train=_fake_train)
    eng._real_ready = True

    res = eng.train(ds, epochs=3, batch=2, imgsz=320, device="cpu")

    assert res["status"] == "ok"
    assert res["model_id"] == "libreyolo-yolox-s"
    assert res["family"] == "libreyolo"
    assert res["variant"] == "yolox-s"
    assert res["dataset_format"] == "yolo"
    assert res["best_checkpoint"].endswith("best.pt")
    assert res["last_checkpoint"].endswith("last.pt")
    assert res["save_dir"] == str(save_dir)
    m = res["metrics"]
    assert m["best_mAP50"] == 0.51
    assert m["best_mAP50_95"] == 0.33
    assert m["best_epoch"] == 7
    assert m["epochs_completed"] == 3  # len(epoch_losses)
    assert m["final_loss"] == 1.23
    assert "training_time_hours" in m
    assert set(res["artifacts"]) == {"weights_dir", "results_csv", "args_yaml"}
    # The trainer received the resolved data.yaml and our safe kwargs.
    assert captured["data"].endswith("data.yaml")
    assert captured["kw"]["epochs"] == 3
    assert captured["kw"]["batch"] == 2
    assert captured["kw"]["allow_download_scripts"] is False


def test_visionmodel_train_unsupported_returns_envelope():
    from visionservex.core.model import VisionModel

    # Standalone HF D-FINE is inference-only: train() returns a structured
    # envelope rather than raising.
    res = VisionModel("dfine-n").train("whatever.yaml")
    assert res["status"] == "TRAINING_NOT_SUPPORTED"
    assert res["model_id"] == "dfine-n"


# ---------------------------------------------------------------------------
# 6. Standalone HF D-FINE stays inference-only; D-FINE via LibreYOLO trains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mid", ["dfine-n", "dfine-s", "dfine-x-o365-coco"])
def test_standalone_dfine_stays_inference_only(mid):
    from visionservex.core.model import _training_capabilities

    cap = _training_capabilities(mid)
    assert cap["train_supported"] is False
    assert cap["finetune_supported"] is False
    assert "HF_BACKEND" in cap["notes"] or "TRAINING_NOT_SUPPORTED" in cap["notes"]


def test_libreyolo_dfine_train_capability():
    """v3.16.0: libreyolo D-FINE *training* is BLOCKED (upstream FDR topk crash);
    it remains inference-ready. Standalone HF ``dfine-*`` also stays inference-only."""
    from visionservex.core.model import _training_capabilities

    cap = _training_capabilities("libreyolo-dfine-n")
    assert cap["train_supported"] is False
    assert cap["exact_blocker"] == "UPSTREAM_DFINE_FDR_TOPK_CRASH"
    assert cap["family"] == "libreyolo"
    assert _training_capabilities("dfine-n")["train_supported"] is False


# ---------------------------------------------------------------------------
# 7. Dataset (YOLO data.yaml) validation
# ---------------------------------------------------------------------------


def test_yolo_dataset_validator(tmp_path):
    from visionservex.data.yolo_dataset import resolve_dataset_yaml, validate_yolo_yaml

    ds = _make_yolo_dataset(tmp_path)
    yaml_path = resolve_dataset_yaml(ds)
    assert yaml_path.name == "data.yaml"

    ok = validate_yolo_yaml(ds)
    assert ok["status"] == "ok"
    assert ok["dataset_format"] == "yolo"
    assert ok["nc"] == 1
    assert ok["names"] == ["obj"]
    assert ok["uses_download_script"] is False

    # Missing 'val' split -> failed verdict with a recorded issue.
    bad = tmp_path / "bad"
    (bad / "images" / "train").mkdir(parents=True)
    (bad / "data.yaml").write_text("train: images/train\nnc: 1\nnames: [obj]\n")
    verdict = validate_yolo_yaml(bad)
    assert verdict["status"] == "failed"
    assert any("val" in i for i in verdict["issues"])


# ---------------------------------------------------------------------------
# 8. Legal: no Ultralytics in the training path
# ---------------------------------------------------------------------------


def test_no_ultralytics_imports():
    src = Path(__file__).resolve().parents[1] / "src" / "visionservex"
    forbidden = ("import ultralytics", "from ultralytics", "ultralytics.YOLO")
    for rel in _TRAINING_PATH_FILES:
        text = (src / rel).read_text()
        for pat in forbidden:
            assert pat not in text, f"{rel} contains forbidden {pat!r}"


# ---------------------------------------------------------------------------
# 9. Docs / capability sync (no overclaiming)
# ---------------------------------------------------------------------------


def test_docs_and_capability_sync():
    docs = Path(__file__).resolve().parents[1] / "docs" / "libreyolo_training.md"
    assert docs.is_file(), "docs/libreyolo_training.md must exist"
    text = docs.read_text().lower()
    assert "yolo" in text  # dataset format documented
    assert "best.pt" in text and "last.pt" in text  # artifacts documented
    assert "onnx" in text  # export documented
    assert "yolo-nas" in text  # exclusion documented
    assert "no ultralytics" in text or "without ultralytics" in text
    # Must state standalone HF D-FINE remains inference-only (no overclaim).
    assert "inference-only" in text or "inference only" in text
