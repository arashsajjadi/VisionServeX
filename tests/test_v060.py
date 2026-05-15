# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tests for v0.6.0: device helpers, new CLI commands, metrics endpoint, downloads audit."""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from visionservex.config import reload_settings


def _img(size=(128, 128)) -> Image.Image:
    return Image.new("RGB", size, color="green")


# ============================================================
# Device helpers
# ============================================================


def test_select_dtype_auto_cpu():
    from visionservex.runtime.device_helpers import select_dtype

    dt = select_dtype("cpu", "auto")
    try:
        import torch

        assert dt == torch.float32
    except ImportError:
        assert dt is None


def test_select_dtype_fp16_cuda():
    pytest.importorskip("torch")
    import torch

    from visionservex.runtime.device_helpers import select_dtype

    assert select_dtype("cuda", "fp16") == torch.float16


def test_move_inputs_no_integer_cast():
    pytest.importorskip("torch")
    import torch

    from visionservex.runtime.device_helpers import move_inputs_to_device

    inputs = {
        "pixel_values": torch.randn(1, 3, 256, 256),
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    out = move_inputs_to_device(inputs, "cpu", torch.float16, cast_floats_only=True)
    assert out["pixel_values"].dtype == torch.float16
    assert out["input_ids"].dtype == torch.long  # MUST NOT be cast
    assert out["attention_mask"].dtype == torch.long  # MUST NOT be cast


def test_device_is_available_cpu():
    from visionservex.runtime.device_helpers import device_is_available

    assert device_is_available("cpu") is True


# ============================================================
# CLI new commands — smoke tests
# ============================================================


def test_gpu_doctor_json(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["gpu", "doctor", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert "cuda" in data or data.get("nvidia_smi_available") is not None


def test_gpu_smoke_test_mock(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(
        app, ["gpu", "smoke-test", "--models", "mock-detect", "--device", "cpu", "--json"]
    )
    assert r.exit_code == 0, r.output
    # Output may have a "Best device:" prefix line before the JSON array
    json_str = r.output
    bracket_pos = json_str.find("[")
    if bracket_pos > 0:
        json_str = json_str[bracket_pos:]
    results = json.loads(json_str)
    assert any(res.get("status") == "ok" for res in results)


def test_benchmark_matrix_mock(monkeypatch, tmp_path):
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
    results = json.loads(r.output)
    assert isinstance(results, list)
    assert results[0]["model_id"] == "mock-detect"
    assert results[0]["status"] == "ok"


def test_parallel_test_mock(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    img_path = tmp_path / "test.jpg"
    _img().save(img_path, "JPEG")

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
            "2",
            "--device",
            "cpu",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert "sequential_p50_ms" in data
    assert "concurrent_wall_p50_ms" in data
    assert "status" in data


def test_downloads_audit_json(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["downloads", "audit", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert "counts" in data
    assert data["counts"]["total"] > 0
    assert "issues" in data


def test_openmmlab_doctor_json(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["openmmlab", "doctor", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert "packages" in data
    assert "mmengine" in data["packages"]


def test_tensorrt_doctor_json(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["tensorrt", "doctor", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert "tensorrt_installed" in data
    assert "onnxruntime_installed" in data


def test_tensorrt_build_dry_run(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    # Create a fake ONNX file
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake onnx")

    runner = CliRunner()
    r = runner.invoke(
        app,
        ["tensorrt", "build", "swinv2-tiny", "--onnx", str(onnx_path), "--dry-run", "--json"],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["dry_run"] is True
    assert "command" in data


# ============================================================
# Prometheus metrics endpoint
# ============================================================


def test_prometheus_metrics_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    s = reload_settings()
    from fastapi.testclient import TestClient

    from visionservex.server.app import create_app

    app = create_app(s)
    client = TestClient(app)
    r = client.get("/metrics/prometheus")
    assert r.status_code == 200
    body = r.text
    assert "visionservex_requests_total" in body
    assert "# HELP" in body
    assert "# TYPE" in body


def test_prometheus_metrics_format(monkeypatch, tmp_path, jpeg_bytes):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    s = reload_settings()
    from fastapi.testclient import TestClient

    from visionservex.server.app import create_app

    app = create_app(s)
    client = TestClient(app)

    # Make a prediction to populate metrics
    r = client.post(
        "/detect",
        files={"image": ("x.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
        data={"model_id": "mock-detect"},
    )
    assert r.status_code == 200

    r2 = client.get("/metrics/prometheus")
    body = r2.text
    # After a prediction, counters should increment
    assert "visionservex_requests_total 1" in body or "visionservex_requests_total" in body


# ============================================================
# Device helpers — device_is_available
# ============================================================


def test_device_is_available_unknown_returns_false():
    from visionservex.runtime.device_helpers import device_is_available

    assert device_is_available("nonexistent_device_abc") is False
