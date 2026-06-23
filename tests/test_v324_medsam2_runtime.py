# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.24: MedSAM2 real-runtime adapter, batch/parallel, training-truth, inputs.

Weight-free + download-free by default. The ONE real-inference test is gated on
``sam2`` being importable AND a checkpoint env var being set, so it skips cleanly
in normal/no-torch CI and runs only in the isolated MedSAM2 env.

Truthfulness guards: MedSAM2 never reports commercial_safe; it is never a runtime
registry model; no VSX training is ever claimed.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

from visionservex.medical import medsam2_runtime as MR
from visionservex.medical import training as MT

pytestmark = pytest.mark.fast


def _sam2_available() -> bool:
    try:
        return importlib.util.find_spec("sam2") is not None
    except Exception:
        return False


# Bind at collection time: the test harness scrubs VISIONSERVEX_* env vars during
# test execution, so re-reading os.environ inside the test body would fail.
_MEDSAM2_CKPT = os.environ.get("VISIONSERVEX_MEDSAM2_CHECKPOINT", "")


# --------------------------------------------------------------------------- #
# Import-light + doctor truth
# --------------------------------------------------------------------------- #
def test_runtime_module_is_import_light():
    # importing the adapter must NOT pull the heavy sam2 stack
    assert "sam2" not in sys.modules


def test_doctor_reports_noncommercial_and_input_modes():
    d = MR.medsam2_doctor()
    assert d["commercial_safe"] is False
    assert d["supported_input_modes"] == ["2d_slice"]
    assert "3d_volume" in d["unsupported_input_modes"] and "video" in d["unsupported_input_modes"]
    # in core CI sam2 is absent → runtime not present + structured code
    if not _sam2_available():
        assert d["runnable_runtime_present"] is False
        assert d["structured_error_code"] == "MEDSAM2_REQUIRED"


def test_runtime_error_to_dict_is_structured_and_noncommercial():
    err = MR.MedSAM2RuntimeError(MR.MEDSAM2_CHECKPOINT_REQUIRED, "no ckpt")
    payload = err.to_dict()
    assert payload["code"] == "MEDSAM2_CHECKPOINT_REQUIRED"
    assert payload["commercial_safe"] is False
    assert payload["status"] == "failed"


@pytest.mark.skipif(
    _sam2_available(), reason="sam2 present → checkpoint-required path covered elsewhere"
)
def test_load_without_deps_raises_required():
    with pytest.raises(MR.MedSAM2RuntimeError) as exc:
        MR.load_medsam2_runtime("/nonexistent/medsam2.pt", device="cpu")
    assert exc.value.code == MR.MEDSAM2_REQUIRED


# --------------------------------------------------------------------------- #
# Input handling (PNG / DICOM-reject / missing)
# --------------------------------------------------------------------------- #
def test_load_2d_input_png(tmp_path):
    import numpy as np
    from PIL import Image

    p = tmp_path / "x.png"
    Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(p)
    arr = MR.load_2d_input(p)
    assert arr.shape == (16, 16, 3)


def test_load_2d_input_dicom_rejected(tmp_path):
    p = tmp_path / "scan.dcm"
    p.write_bytes(b"not really dicom")
    with pytest.raises(MR.MedSAM2RuntimeError) as exc:
        MR.load_2d_input(p)
    assert exc.value.code == MR.MEDSAM2_UNSUPPORTED_INPUT
    assert "DICOM" in str(exc.value)


def test_load_2d_input_missing(tmp_path):
    with pytest.raises(MR.MedSAM2RuntimeError) as exc:
        MR.load_2d_input(tmp_path / "nope.png")
    assert exc.value.code == MR.MEDSAM2_UNSUPPORTED_INPUT


# --------------------------------------------------------------------------- #
# Training truth (no fake training)
# --------------------------------------------------------------------------- #
def test_training_matrix_has_no_vsx_trainable_models():
    for mid, row in MT.TRAINING_MATRIX.items():
        assert row["trainable_in_vsx"] is False, mid
        assert row["finetunable_in_vsx"] is False, mid
    assert MT.TRAINING_MATRIX["medsam"]["status"] == "NOT_TRAINABLE_BY_DESIGN"
    assert MT.TRAINING_MATRIX["medsam2"]["status"] == "EXTERNAL_TRAINING_ONLY"


def test_train_doctor_claims_no_vsx_training():
    d = MT.train_doctor()
    assert d["vsx_trains_any_model"] is False


def test_dataset_validator_accepts_good_pairs(tmp_path):
    import numpy as np
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    for i in range(3):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / "images" / f"s{i}.png")
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / "masks" / f"s{i}.png")
    rep = MT.validate_segmentation_dataset(tmp_path)
    assert rep["valid"] is True
    assert rep["n_pairs"] == 3


def test_dataset_validator_flags_missing_mask(tmp_path):
    import numpy as np
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / "images" / "a.png")
    rep = MT.validate_segmentation_dataset(tmp_path)
    assert rep["valid"] is False
    assert any("without a matching mask" in e for e in rep["errors"])


def test_dry_run_generates_commands_but_does_not_train(tmp_path):
    out = MT.dry_run("nnunet", tmp_path, tmp_path / "out")
    assert out["status"] == "dry_run"
    assert out["trains_in_vsx"] is False
    assert any("nnUNetv2" in c for c in out["generated_commands"])


# --------------------------------------------------------------------------- #
# Parallel executor: order stability + failure isolation + strict stop
# --------------------------------------------------------------------------- #
def test_parallel_executor_preserves_order_out_of_order_completion():
    import time

    from visionservex.medical.parallel import run_ordered

    def fn(item, idx):
        time.sleep(0.02 * (3 - idx % 4))  # later items finish sooner
        return {"v": item}

    items = list(range(6))
    res = run_ordered(items, fn, workers=4, continue_on_error=True)
    assert [r.index for r in res] == items
    assert [r.value["v"] for r in res] == items


def test_parallel_executor_isolates_per_item_failure():
    from visionservex.medical.parallel import run_ordered

    def fn(item, idx):
        if item == 2:
            raise ValueError("boom")
        return {"v": item}

    res = run_ordered(list(range(4)), fn, workers=3, continue_on_error=True)
    statuses = [r.status for r in res]
    assert statuses == ["ok", "ok", "failed", "ok"]
    assert res[2].error and "boom" in res[2].error


def test_parallel_executor_strict_stops_and_skips_remainder():
    from visionservex.medical.parallel import run_ordered

    def fn(item, idx):
        if item == 1:
            raise ValueError("stop here")
        return {"v": item}

    res = run_ordered(list(range(4)), fn, workers=1, strict=True)
    assert [r.status for r in res] == ["ok", "failed", "skipped", "skipped"]


# --------------------------------------------------------------------------- #
# Batch: deterministic naming + order + manifest + no-overwrite (mocked predictor)
# --------------------------------------------------------------------------- #
def _fake_seg_result():
    import numpy as np

    from visionservex.core.results import Box, Segment, SegmentationResult

    m = np.zeros((8, 8), np.uint8)
    m[2:6, 2:6] = 1
    seg = Segment(box=Box(2, 2, 6, 6), score=0.9, label="lesion", mask=m)
    r = SegmentationResult(
        kind="segmentation",
        model_id="medsam2",
        task="foundation_segment",
        image_size=(8, 8),
        device="cpu",
        precision="fp32",
        backend="medsam2_runtime",
        segments=[seg],
    )
    return r


def test_batch_naming_order_and_manifest(tmp_path, monkeypatch):
    import numpy as np

    from visionservex.medical import medsam2_batch as MB

    monkeypatch.setattr(
        MB,
        "load_medsam2_runtime",
        lambda *a, **k: type("RT", (), {"config_path": "cfg", "load_time_seconds": 0.0})(),
    )
    monkeypatch.setattr(MB, "load_2d_input", lambda p: np.zeros((8, 8, 3), np.uint8))
    monkeypatch.setattr(MB, "segment_2d", lambda rt, img, **k: _fake_seg_result())

    inputs = [str(tmp_path / "a.png"), str(tmp_path / "b.png")]
    out = tmp_path / "out"
    man = MB.run_medsam2_batch(inputs, checkpoint="x.pt", out_dir=out, device="cpu")
    assert man["n_inputs"] == 2 and man["n_ok"] == 2
    assert [it["index"] for it in man["items"]] == [0, 1]
    assert man["commercial_safe"] is False
    assert (out / "00000_a_medsam2_mask_000.png").exists()
    assert (out / "00001_b_medsam2_mask_000.png").exists()
    assert (out / "medsam2_batch_manifest.json").exists()

    # no overwrite without the flag
    man2 = MB.run_medsam2_batch(
        inputs, checkpoint="x.pt", out_dir=out, device="cpu", overwrite=False
    )
    assert man2["n_failed"] == 2
    assert all(it["error_code"] == "OUTPUT_EXISTS" for it in man2["items"])


def test_batch_gpu_default_does_not_duplicate_model(tmp_path, monkeypatch):
    import numpy as np

    from visionservex.medical import medsam2_batch as MB

    loads = {"n": 0}

    def _fake_load(*a, **k):
        loads["n"] += 1
        return type("RT", (), {"config_path": "cfg", "load_time_seconds": 0.0})()

    monkeypatch.setattr(MB, "load_medsam2_runtime", _fake_load)
    monkeypatch.setattr(MB, "load_2d_input", lambda p: np.zeros((8, 8, 3), np.uint8))
    monkeypatch.setattr(MB, "segment_2d", lambda rt, img, **k: _fake_seg_result())

    man = MB.run_medsam2_batch(
        [str(tmp_path / f"{i}.png") for i in range(3)],
        checkpoint="x.pt",
        out_dir=tmp_path / "o",
        device="cuda",
        workers=4,
    )
    assert loads["n"] == 1  # model loaded ONCE, never duplicated
    assert man["effective_workers"] == 1
    assert any("clamped" in w for w in man["warnings"])


# --------------------------------------------------------------------------- #
# CLI smoke (core env)
# --------------------------------------------------------------------------- #
def test_cli_medsam2_doctor_and_train():
    import json

    from typer.testing import CliRunner

    from visionservex.cli.medical_commands import app

    r = CliRunner()
    o = r.invoke(app, ["medsam2", "doctor", "--json"])
    assert o.exit_code == 0
    assert json.loads(o.stdout)["commercial_safe"] is False

    o = r.invoke(app, ["train", "doctor", "--json"])
    assert o.exit_code == 0
    assert json.loads(o.stdout)["vsx_trains_any_model"] is False


def test_cli_medsam2_load_missing_is_structured():
    import json

    from typer.testing import CliRunner

    from visionservex.cli.medical_commands import app

    o = CliRunner().invoke(app, ["medsam2", "load", "--checkpoint", "/no/such.pt", "--json"])
    assert o.exit_code != 0
    assert json.loads(o.stdout)["code"] in {"MEDSAM2_REQUIRED", "MEDSAM2_CHECKPOINT_REQUIRED"}


# --------------------------------------------------------------------------- #
# Registry truth: MedSAM2 is NOT a runtime registry model (no false claim)
# --------------------------------------------------------------------------- #
def test_medsam2_still_not_a_runtime_registry_model():
    from visionservex.registry import RegistryError, default_registry

    with pytest.raises(RegistryError):
        default_registry().get("medsam2")


# --------------------------------------------------------------------------- #
# REAL inference — gated; runs only in the isolated MedSAM2 env with a checkpoint
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.real_model
@pytest.mark.sidecar
@pytest.mark.skipif(
    not _sam2_available() or not _MEDSAM2_CKPT,
    reason="needs sam2 installed + VISIONSERVEX_MEDSAM2_CHECKPOINT pointing at a real .pt",
)
def test_real_medsam2_2d_inference_produces_mask():
    import numpy as np

    rt = MR.load_medsam2_runtime(_MEDSAM2_CKPT, device="cpu")
    img = np.zeros((256, 256, 3), np.uint8)
    img[64:192, 64:192] = 200
    result = MR.segment_2d(rt, img, boxes=[[64, 64, 192, 192]])
    assert len(result.segments) == 1
    seg = result.segments[0]
    assert seg.mask.shape == (256, 256)
    assert int(seg.mask.sum()) > 0  # real, non-empty mask
    assert result.metadata["commercial_safe"] is False
