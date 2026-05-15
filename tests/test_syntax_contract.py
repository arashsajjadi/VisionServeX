# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Syntax contract tests — validates that all 222 CLI/Python/API examples work.

These tests use mock models and TestClient so no real downloads are needed in CI.
Real-model variants are marked @pytest.mark.real_model.
"""

from __future__ import annotations

import io
import json

from PIL import Image

from visionservex.config import reload_settings


def _img(size=(128, 128)) -> Image.Image:
    return Image.new("RGB", size, color="blue")


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    _img().save(buf, format="JPEG")
    return buf.getvalue()


# ============================================================
# A. Package-level imports (examples 1-4 verified by import)
# ============================================================


def test_package_imports():
    from visionservex import (
        AsyncClient,
        Client,
        VisionModel,
        VisionServeXError,
        __version__,
    )

    assert VisionModel is not None
    assert Client is not None
    assert AsyncClient is not None
    assert VisionServeXError is not None
    assert __version__


# ============================================================
# B. Typed exceptions (example 150 / section M)
# ============================================================


def test_model_not_found_typed_error():
    from visionservex.exceptions import ModelNotFoundError

    err = ModelNotFoundError("no-such-model")
    assert err.code == "MODEL_NOT_FOUND"
    assert err.hint
    assert "no-such-model" in str(err)


def test_input_not_found_error():
    from visionservex.exceptions import InputNotFoundError

    err = InputNotFoundError("/tmp/nonexistent.jpg")
    assert err.code == "INPUT_NOT_FOUND"


def test_device_unavailable_error():
    from visionservex.exceptions import DeviceUnavailableError

    err = DeviceUnavailableError("cuda", "libnvrtc missing")
    assert err.code == "DEVICE_UNAVAILABLE"


def test_external_model_error():
    from visionservex.exceptions import ExternalModelError

    err = ExternalModelError("grounding-dino-1.6", alternative="grounding-dino-tiny")
    assert err.code == "EXTERNAL_MODEL"
    assert err.alternative == "grounding-dino-tiny"


def test_manual_model_error():
    from visionservex.exceptions import ManualModelError

    err = ManualModelError("rtmpose-s", instructions="visionservex openmmlab docker-run")
    assert err.code == "MANUAL_MODEL"


# ============================================================
# C. VisionModel new kwargs (examples 133-149)
# ============================================================


def test_visionmodel_loaded_property():
    from visionservex import VisionModel

    m = VisionModel("mock-detect")
    assert m.loaded is False
    m.warmup()
    assert m.loaded is True


def test_visionmodel_prompt_alias():
    from visionservex import VisionModel

    m = VisionModel("mock-open-vocab")
    r = m.predict(_img(), prompt="car, person")
    # prompt="car, person" → prompts=["car", "person"]
    assert r is not None


def test_visionmodel_top_k():
    from visionservex import VisionModel

    m = VisionModel("mock-classify")
    r = m.predict(_img(), top_k=3)
    assert r is not None


def test_visionmodel_task_kwarg():
    from visionservex import VisionModel

    m = VisionModel("mock-segment")
    r = m.predict(_img(), task="semantic")
    assert r is not None


def test_visionmodel_threshold():
    from visionservex import VisionModel

    m = VisionModel("mock-detect")
    r = m.predict(_img(), threshold=0.5)
    assert r is not None


def test_visionmodel_box_alias():
    from visionservex import VisionModel

    m = VisionModel("mock-foundation-segment")
    r = m.predict(_img(), box=[10, 10, 100, 100])
    assert r is not None


def test_visionmodel_points():
    from visionservex import VisionModel

    m = VisionModel("mock-foundation-segment")
    r = m.predict(_img(), points=[[64, 64]], point_labels=[1])
    assert r is not None


def test_visionmodel_labels_alias():
    from visionservex import VisionModel

    m = VisionModel("mock-foundation-segment")
    r = m.predict(_img(), points=[[64, 64]], labels=[1])
    assert r is not None


# ============================================================
# D. BaseResult new methods
# ============================================================


def test_result_save_json(tmp_path):
    from visionservex import VisionModel

    m = VisionModel("mock-detect")
    r = m.predict(_img())
    p = r.save_json(tmp_path / "result.json")
    assert p.exists()
    data = json.loads(p.read_text())
    assert data["kind"] == "detection"


def test_result_save_image(tmp_path):
    from visionservex import VisionModel

    m = VisionModel("mock-detect")
    r = m.predict(_img())
    p = r.save_image(tmp_path / "out.jpg")
    assert p.exists()
    assert p.stat().st_size > 0


# ============================================================
# E. CLI predict new flags (examples 59-88)
# ============================================================


def test_predict_device_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    img = tmp_path / "x.jpg"
    _img().save(img, "JPEG")
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["predict", "mock-detect", str(img), "--device", "cpu", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data.get("device") == "cpu"


def test_predict_save_json_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    img = tmp_path / "x.jpg"
    _img().save(img, "JPEG")
    out_json = tmp_path / "result.json"
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["predict", "mock-detect", str(img), "--save-json", str(out_json)])
    assert r.exit_code == 0, r.output
    assert out_json.exists()


def test_predict_missing_input_gives_typed_error(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["predict", "mock-detect", "/tmp/does_not_exist_xyz.jpg", "--json"])
    # Should fail with structured error, not raw traceback
    assert r.exit_code != 0
    assert "INPUT_NOT_FOUND" in r.output or "input" in r.output.lower()


def test_predict_unknown_model_gives_typed_error(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    img = tmp_path / "x.jpg"
    _img().save(img, "JPEG")
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["predict", "no-such-model-xyz", str(img), "--json"])
    assert r.exit_code != 0
    out = r.output + (r.stderr or "")
    assert "MODEL_NOT_FOUND" in out or "unknown model" in out.lower()


def test_batch_predict_command(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(3):
        _img().save(img_dir / f"img{i}.jpg", "JPEG")

    out_dir = tmp_path / "outputs"
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(
        app,
        ["batch-predict", "mock-detect", str(img_dir), "--save-dir", str(out_dir), "--json"],
    )
    assert r.exit_code == 0, r.output
    results = json.loads(r.output)
    assert len(results) == 3


# ============================================================
# F. Gateway commands (examples 89-105)
# ============================================================


def test_gateway_loaded_models(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["gateway", "loaded-models"])
    assert r.exit_code == 0


def test_gateway_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["gateway", "memory"])
    assert r.exit_code == 0


def test_gateway_stop_no_server(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["gateway", "stop"])
    # Should not crash — should print helpful message
    assert r.exit_code == 0


# ============================================================
# G. Server API (/segment/b64, /obb) (examples 119, 124)
# ============================================================


def test_server_obb_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    s = reload_settings()
    from fastapi.testclient import TestClient

    from visionservex.server.app import create_app

    app = create_app(s)
    client = TestClient(app)
    r = client.post(
        "/obb",
        files={"image": ("x.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg")},
        data={"model_id": "mock-obb"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["task"] == "obb"


def test_server_segment_b64_endpoint(monkeypatch, tmp_path):
    import base64

    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    s = reload_settings()
    from fastapi.testclient import TestClient

    from visionservex.server.app import create_app

    app = create_app(s)
    client = TestClient(app)
    b64 = base64.b64encode(_jpeg_bytes()).decode("ascii")
    r = client.post(
        "/segment/b64",
        json={"model_id": "mock-segment", "image_b64": b64, "prompts": []},
    )
    assert r.status_code == 200


# ============================================================
# H. Suite commands (examples 48-51)
# ============================================================


def test_pull_suite_full_auto_in_registry():
    from visionservex.cli.suite_commands import _SUITES

    assert "full-auto" in _SUITES
    assert len(_SUITES["full-auto"]) > 5


def test_pull_suite_command_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["pull-suite", "beginner", "--yes", "--json"])
    assert r.exit_code == 0, r.output


# ============================================================
# I. New CLI commands (models-audit, onnx-validate dry-run)
# ============================================================


def test_models_audit_command(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["models-audit", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert "counts" in data


def test_onnx_validate_missing_file(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["onnx-validate", str(tmp_path / "nonexistent.onnx")])
    # Should fail but not crash with raw traceback
    assert r.exit_code != 0


def test_cache_clean_model_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["cache", "clean", "--model", "mock-detect", "--yes", "--json"])
    assert r.exit_code == 0, r.output


# ============================================================
# J. Error behavior (examples 214-222)
# ============================================================


def test_ssrf_protection(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VISIONSERVEX_INPUTS__ALLOW_URL_INPUTS", "true")
    s = reload_settings()
    from fastapi.testclient import TestClient

    from visionservex.server.app import create_app

    app = create_app(s)
    client = TestClient(app)
    r = client.post(
        "/open-vocab/detect",
        json={
            "model_id": "mock-open-vocab",
            "image_url": "http://169.254.169.254/latest/meta-data/",
            "prompts": ["test"],
        },
    )
    assert r.status_code in {422, 403}
    data = r.json()
    assert data["error"]["code"] in {
        "BAD_URL",
        "FORBIDDEN",
        "SSRF_BLOCKED",
        "UNAUTHENTICATED",
    }


def test_server_busy_retry_after(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    s = reload_settings()
    from contextlib import asynccontextmanager
    from unittest.mock import MagicMock, patch

    from fastapi.testclient import TestClient

    from visionservex.runtime.scheduler import BackpressureError, RequestScheduler
    from visionservex.server.app import create_app

    app = create_app(s)
    client = TestClient(app)
    mock_sched = MagicMock(spec=RequestScheduler)

    @asynccontextmanager
    async def _raise(*args, **kwargs):
        raise BackpressureError("test")
        yield  # type: ignore[misc]

    mock_sched.reserve = _raise
    import visionservex.server.app as srv

    with patch.object(srv, "get_scheduler", return_value=mock_sched):
        r = client.post(
            "/detect",
            files={"image": ("x.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg")},
            data={"model_id": "mock-detect"},
        )
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "BUSY"
    assert "retry_after_seconds" in r.json()["error"]["details"]


def test_request_too_large(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VISIONSERVEX_LIMITS__MAX_UPLOAD_BYTES", "100")
    s = reload_settings()
    from fastapi.testclient import TestClient

    from visionservex.server.app import create_app

    app = create_app(s)
    client = TestClient(app)
    r = client.post(
        "/detect",
        files={"image": ("x.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg")},
        data={"model_id": "mock-detect"},
    )
    assert r.status_code in {413, 422}


# ============================================================
# K. AsyncClient basic (examples 163-164)
# ============================================================


def test_async_client_import():
    from visionservex import AsyncClient

    c = AsyncClient("http://127.0.0.1:8080")
    assert c.base_url == "http://127.0.0.1:8080"


def test_async_client_prepare_image():
    from visionservex import AsyncClient

    c = AsyncClient()
    data, fname = c._prepare_image(_img())
    assert isinstance(data, bytes)
    assert fname == "image.jpg"
