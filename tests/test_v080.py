# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tests for v0.8.0: Client, gateway commands, pull-suite, scheduler, SSE."""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from visionservex.config import reload_settings
from visionservex.registry import default_registry


def _img(size=(128, 128)) -> Image.Image:
    return Image.new("RGB", size, color="gray")


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    _img().save(buf, format="JPEG")
    return buf.getvalue()


# ============================================================
# Client class
# ============================================================


def test_client_import():
    from visionservex import Client, ClientResult, GatewayError

    assert Client is not None
    assert ClientResult is not None
    assert GatewayError is not None


def test_client_prepare_image_bytes():
    from visionservex.client import Client

    c = Client()
    data, _fname = c._prepare_image(_jpeg_bytes())
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_client_prepare_image_pil():
    from visionservex.client import Client

    c = Client()
    data, fname = c._prepare_image(_img())
    assert isinstance(data, bytes)
    assert fname == "image.jpg"


def test_client_result_properties():
    from visionservex.client import ClientResult

    r = ClientResult(
        {
            "model_id": "dfine-n",
            "task": "detect",
            "device": "cuda",
            "precision": "fp32",
            "backend": "huggingface_dfine",
            "latency_ms": 8.5,
            "results": [{"label": "car"}],
            "warnings": [],
            "request_id": "abc",
            "status": "completed",
        }
    )
    assert r.model_id == "dfine-n"
    assert r.task == "detect"
    assert r.device == "cuda"
    assert r.latency_ms == pytest.approx(8.5)
    assert len(r.results) == 1
    assert "dfine-n" in repr(r)


def test_gateway_error_message():
    from visionservex.client import GatewayError

    e = GatewayError("MODEL_MISSING", "weights not found", hint="run pull", details={"x": 1})
    assert "MODEL_MISSING" in str(e)
    assert e.code == "MODEL_MISSING"
    assert e.hint == "run pull"


# ============================================================
# Gateway server endpoints
# ============================================================


def test_gateway_status_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    s = reload_settings()
    from fastapi.testclient import TestClient

    from visionservex.server.app import create_app

    app = create_app(s)
    client = TestClient(app)
    r = client.get("/gateway/status")
    assert r.status_code == 200
    data = r.json()
    assert "version" in data
    assert "best_device" in data
    assert "loaded_models" in data
    assert "scheduler" in data


def test_gateway_warmup_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    s = reload_settings()
    from fastapi.testclient import TestClient

    from visionservex.server.app import create_app

    app = create_app(s)
    client = TestClient(app)
    r = client.post("/gateway/warmup", json=["mock-detect"])
    assert r.status_code == 200
    data = r.json()
    assert "warmed_up" in data


def test_job_events_snapshot(monkeypatch, tmp_path):
    """SSE endpoint without ?sse=true returns a snapshot dict."""
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    s = reload_settings()
    from fastapi.testclient import TestClient

    from visionservex.server.app import create_app

    app = create_app(s)
    client = TestClient(app)
    r = client.get("/jobs/nonexistent/events")
    assert r.status_code == 404


# ============================================================
# Gateway CLI commands
# ============================================================


def test_gateway_doctor_json(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["gateway", "doctor", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert "best_device" in data
    assert "auto_pull" in data


def test_gateway_profile_shows_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["gateway", "profile", "laptop", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert "VISIONSERVEX_SERVER__HOST" in data


def test_gateway_openapi_output(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["gateway", "openapi"])
    assert r.exit_code == 0
    assert "openapi.json" in r.output


# ============================================================
# pull-suite
# ============================================================


def test_suite_list_json(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["suite", "list", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert "beginner" in data
    assert "gpu-demo" in data
    assert "dfine-n" in data["beginner"]


def test_pull_suite_alias_exists():
    from visionservex.cli.main import app

    command_names = [c.name for c in app.registered_commands]
    assert "pull-suite" in command_names


# ============================================================
# Scheduler policies
# ============================================================


def test_scheduler_profile_json(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["scheduler", "profile", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert "dfine-n" in data
    assert data["dfine-n"]["policy"] == "queue_recommended"
    assert data["swinv2-tiny"]["policy"] == "acceptable_parallelism"


def test_scheduler_recommend_json(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["scheduler", "recommend", "--model", "dfine-n", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["recommended_max_concurrency"] == 1
    assert "queue" in data["recommended_policy"]


def test_get_model_max_concurrency():
    from visionservex.cli.suite_commands import get_model_max_concurrency

    assert get_model_max_concurrency("dfine-n") == 1
    assert get_model_max_concurrency("swinv2-tiny") == 2
    assert get_model_max_concurrency("unknown-model") == 2  # default


# ============================================================
# Model status completeness audit
# ============================================================


def test_every_wired_model_has_implementation_status():
    reg = default_registry()
    for e in reg.list():
        assert e.implementation_status in {
            "wired",
            "partial",
            "stub",
        }, f"{e.id}: invalid implementation_status"


def test_no_vague_status_without_engine():
    """Models that are 'stub' must have a reason (notes or warnings)."""
    reg = default_registry()
    for e in reg.list():
        if e.implementation_status == "stub" and e.status not in {"external"}:
            # Stub models must have at least an engine entry or notes
            assert e.engine or e.notes or e.warnings, (
                f"{e.id}: stub without engine/notes/warnings is too vague"
            )
