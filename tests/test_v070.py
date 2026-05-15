# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tests for v0.7.0: CUDA fix, fp32 auto-precision, sidecar engine, downloads."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from visionservex.config import reload_settings
from visionservex.registry import default_registry


def _img(size=(128, 128)) -> Image.Image:
    return Image.new("RGB", size, color="green")


# ============================================================
# NVRTC path patching
# ============================================================


def test_nvrtc_patch_does_not_crash():
    """Importing device module should not raise even if libnvrtc dirs are absent."""
    import os

    import visionservex.runtime.device as dmod  # noqa: F401 — side-effect test

    ld = os.environ.get("LD_LIBRARY_PATH", "")
    # LD_LIBRARY_PATH should be a string, not raise
    assert isinstance(ld, str)


# ============================================================
# fp32 auto-precision fix
# ============================================================


def test_auto_precision_is_fp32():
    """precision='auto' must now default to fp32 for safety."""
    from visionservex.core.model import VisionModel

    m = VisionModel("mock-detect", device="cpu")
    assert m.precision == "fp32", f"Expected fp32, got {m.precision}"


def test_explicit_fp16_falls_back_gracefully():
    """fp16 on a model that only supports fp32 falls back to fp32 (correct)."""
    from visionservex.core.model import VisionModel

    # mock-detect only supports fp32; requesting fp16 falls back to first supported
    m = VisionModel("mock-detect", device="cpu", precision="fp16")
    assert m.precision == "fp32"  # fallback to first supported precision


def test_explicit_fp16_on_capable_model():
    """Models that declare fp16 support accept it."""
    from visionservex.core.model import VisionModel

    # dfine-s supports fp32 and fp16
    m = VisionModel("dfine-s", device="cpu", precision="fp16")
    assert m.precision == "fp16"


# ============================================================
# OpenMMLab sidecar engine registry
# ============================================================


def test_rtmpose_s_uses_sidecar_engine():
    e = default_registry().get("rtmpose-s")
    assert e.engine == "openmmlab_sidecar"
    assert e.implementation_status == "partial"


def test_rtmdet_r2_s_uses_sidecar_engine():
    e = default_registry().get("rtmdet-r2-s")
    assert e.engine == "openmmlab_sidecar"
    assert e.implementation_status == "partial"


def test_sidecar_engine_raises_without_sidecar(monkeypatch, tmp_path):
    """Without a running sidecar, load() must raise MissingDependencyError."""
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VISIONSERVEX_MODELS__ALLOW_MOCK_FALLBACK", "false")
    monkeypatch.setenv("VISIONSERVEX_OPENMMLAB_SIDECAR_URL", "http://127.0.0.1:19999")
    reload_settings()

    from visionservex.engines.base import MissingDependencyError
    from visionservex.engines.openmmlab_sidecar import OpenMMLabSidecarEngine

    e = default_registry().get("rtmpose-s")
    engine = OpenMMLabSidecarEngine(e)
    with pytest.raises(MissingDependencyError):
        engine.load(device="cpu", precision="fp32")


def test_sidecar_health_false_for_unreachable(monkeypatch):
    """_sidecar_health must return False for an unreachable URL."""
    monkeypatch.setenv("VISIONSERVEX_OPENMMLAB_SIDECAR_URL", "http://127.0.0.1:19999")
    from visionservex.engines.openmmlab_sidecar import _sidecar_health

    result = _sidecar_health()
    assert result is False


# ============================================================
# GPU benchmark smoke (no real weights needed — uses mock-detect on CPU)
# ============================================================


def test_benchmark_matrix_cuda_key_in_json(monkeypatch, tmp_path):
    """benchmark-matrix --json output has correct schema fields."""
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "benchmark-matrix",
            "--models",
            "mock-detect",
            "--devices",
            "cpu",
            "--runs",
            "2",
            "--warmup",
            "1",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    entry = data[0]
    assert entry["status"] == "ok"
    assert "warm_p50_ms" in entry
    assert "throughput_req_s" in entry
    assert "fallback_reason" in entry


def test_parallel_test_result_schema(monkeypatch, tmp_path):
    """parallel-test --json must have slowdown and status fields."""
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()

    img_path = tmp_path / "test.jpg"
    _img().save(img_path, "JPEG")

    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "parallel-test",
            "mock-detect",
            str(img_path),
            "--concurrency",
            "2",
            "--runs",
            "3",
            "--device",
            "cpu",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert "slowdown_pct" in data
    assert "status" in data
    valid_statuses = {
        "excellent_parallelism",
        "acceptable_parallelism",
        "scheduler_needs_queueing",
        "protected_throughput",
    }
    assert data["status"] in valid_statuses


# ============================================================
# Downloads audit: 0 missing required metadata
# ============================================================


def test_downloads_audit_no_missing_required(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["downloads", "audit", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["counts"]["missing_required"] == 0, (
        f"Expected 0 missing required, got: {data['counts']['missing_required']}"
    )


# ============================================================
# GPU-specific tests (opt-in)
# ============================================================


@pytest.mark.gpu
def test_dfine_n_cuda_inference():
    from visionservex import VisionModel

    m = VisionModel("dfine-n", device="cuda", precision="fp32")
    r = m.predict(_img())
    assert r.device == "cuda", f"Expected cuda, got {r.device}"
    assert r.precision == "fp32"
    assert r.backend == "huggingface_dfine"
    assert r.fallback_reason is None


@pytest.mark.gpu
def test_swinv2_tiny_cuda_inference():
    from visionservex import VisionModel

    m = VisionModel("swinv2-tiny", device="cuda", precision="fp32")
    r = m.predict(_img(size=(256, 256)))
    assert r.device == "cuda"
    assert len(r.top_k) > 0


@pytest.mark.gpu
def test_sam2_hiera_tiny_cuda():
    from visionservex import VisionModel

    m = VisionModel("sam2-hiera-tiny", device="cuda", precision="fp32")
    r = m.predict(_img(), points=[[64, 64]], point_labels=[1])
    assert r.device == "cuda"
    assert len(r.segments) >= 1


@pytest.mark.gpu
def test_grounding_dino_cuda_fp32():
    from visionservex import VisionModel

    m = VisionModel("grounding-dino-tiny", device="cuda", precision="fp32")
    r = m.predict(_img(size=(320, 240)), prompts=["object"])
    assert r.device == "cuda"
    assert r.precision == "fp32"
