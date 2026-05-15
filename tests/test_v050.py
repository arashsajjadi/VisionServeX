# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tests for v0.5.0 features: device sanity, concurrency, grounded-sam2, ONNX export."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from visionservex.config import reload_settings
from visionservex.registry import default_registry


def _img(size=(128, 128), color="blue") -> Image.Image:
    return Image.new("RGB", size, color=color)


# ============================================================
# Device module — sanity check + multi-GPU
# ============================================================


def test_device_info_has_sanity_fields():
    from visionservex.runtime.device import available_devices

    devs = available_devices()
    cpu = next(d for d in devs if d.name == "cpu")
    assert cpu.sanity_ok is True
    assert cpu.available is True


def test_cpu_always_available():
    from visionservex.runtime.device import best_device, resolve_device

    bd = best_device(supported=["cpu"])
    assert bd.name == "cpu"
    res = resolve_device(preference="cpu", supported=["cpu", "cuda"])
    assert res == "cpu"


def test_broken_cuda_not_selected():
    """If CUDA sanity fails, auto-selection must not pick it."""
    import visionservex.runtime.device as dmod
    from visionservex.runtime.device import DeviceInfo, resolve_device

    broken_cuda = DeviceInfo(
        name="cuda",
        available=True,
        detail="GPU detected but broken",
        sanity_ok=False,
        sanity_error="test error",
    )
    DeviceInfo(name="cpu", available=True, sanity_ok=True)

    with (
        patch.object(dmod, "_all_cuda_devices", return_value=[broken_cuda]),
        patch.object(dmod, "_mps_info", return_value=DeviceInfo("mps", False)),
    ):
        selected = resolve_device(preference="auto", supported=["cpu", "cuda"])
        assert selected == "cpu", f"Expected CPU fallback, got {selected}"


def test_device_benchmark_cpu():
    from visionservex.runtime.device import device_benchmark

    result = device_benchmark("cpu", quick=True)
    assert result["ok"] is True
    assert result["avg_ms"] > 0
    assert result["throughput_gflops"] > 0


def test_device_to_dict_includes_sanity():
    from visionservex.runtime.device import available_devices

    for d in available_devices():
        data = d.to_dict()
        assert "sanity_ok" in data
        assert "sanity_error" in data


# ============================================================
# Registry — grounded-sam2
# ============================================================


def test_grounded_sam2_wired():
    e = default_registry().get("grounded-sam2")
    assert e.implementation_status == "wired"
    assert e.engine == "grounded_sam2"
    assert e.status == "beta"
    assert e.auto_download is True
    assert e.install_extra == "hf"


def test_grounded_sam2_metadata_has_submodels():
    e = default_registry().get("grounded-sam2")
    # Implementation notes should mention composition
    assert e.implementation_notes is not None
    assert "SAM" in e.implementation_notes


# ============================================================
# Concurrency — scheduler and backpressure
# ============================================================


def test_retry_after_in_busy_error_details(monkeypatch, tmp_path):
    """The BUSY error details must include retry_after_seconds."""
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VISIONSERVEX_RUNTIME__SERVER_BUSY_RETRY_AFTER_S", "3")
    s = reload_settings()
    from contextlib import asynccontextmanager

    from fastapi.testclient import TestClient

    from visionservex.runtime.scheduler import BackpressureError, RequestScheduler
    from visionservex.server.app import create_app

    app = create_app(s)
    client = TestClient(app)

    buf = io.BytesIO()
    _img().save(buf, "JPEG")

    # Patch the scheduler so reserve() raises BackpressureError
    mock_sched = MagicMock(spec=RequestScheduler)

    @asynccontextmanager
    async def _raise(*args, **kwargs):
        raise BackpressureError("test queue full")
        yield  # type: ignore[misc]

    mock_sched.reserve = _raise

    import visionservex.server.app as srv_app

    with patch.object(srv_app, "get_scheduler", return_value=mock_sched):
        buf.seek(0)
        r = client.post(
            "/detect",
            files={"image": ("x.jpg", buf, "image/jpeg")},
            data={"model_id": "mock-detect"},
        )
    assert r.status_code == 503
    body = r.json()
    assert body["error"]["code"] == "BUSY"
    assert "retry_after_seconds" in body["error"]["details"]


def test_config_has_new_runtime_fields():
    s = reload_settings()
    assert hasattr(s.runtime, "max_global_concurrency")
    assert hasattr(s.runtime, "prefer_fastest_device")
    assert hasattr(s.runtime, "gpu_sanity_check")
    assert hasattr(s.runtime, "server_busy_retry_after_s")
    assert s.runtime.max_global_concurrency > 0
    assert s.runtime.server_busy_retry_after_s >= 1


# ============================================================
# ONNX export
# ============================================================


@pytest.mark.real_model
def test_swinv2_onnx_export(tmp_path):
    pytest.importorskip("torch")
    try:
        import onnxscript  # noqa: F401
    except ImportError:
        pytest.skip("onnxscript not installed")

    from visionservex import VisionModel

    m = VisionModel("swinv2-tiny", device="cpu")
    m.warmup()
    out = tmp_path / "swinv2.onnx"
    path = m.export(format="onnx", output_path=out)
    assert path.exists()
    assert path.stat().st_size > 100_000, "ONNX file seems too small"
    # Cleanup — exported models must not be committed to git
    path.unlink()


def test_export_unsupported_format_raises():
    pytest.importorskip("transformers")
    from visionservex import VisionModel

    m = VisionModel("swinv2-tiny", device="cpu")
    with pytest.raises(NotImplementedError):
        m.export(format="tensorrt", output_path="/tmp/test.engine")


# ============================================================
# Grounded-SAM2 engine real smoke test
# ============================================================


@pytest.mark.real_model
def test_grounded_sam2_real_pipeline():
    pytest.importorskip("transformers")
    from visionservex import VisionModel

    m = VisionModel("grounded-sam2", device="cpu", precision="fp32")
    r = m.predict(_img(size=(256, 256), color="green"), prompts=["green area"])
    assert r.kind == "segmentation"
    assert r.backend == "composed_gd_sam2"
    assert r.metadata.get("detector_model_id") == "grounding-dino-tiny"
    assert r.metadata.get("segmenter_model_id") == "sam2-hiera-tiny"
    assert hasattr(r, "segments")
