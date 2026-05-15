# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tests for auto-pull behavior in the FastAPI server."""

from __future__ import annotations

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from visionservex.config import reload_settings
from visionservex.server.app import create_app


def _client(env: dict, monkeypatch) -> TestClient:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    s = reload_settings()
    return TestClient(create_app(s))


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_missing_model_without_autopull_returns_clear_error(monkeypatch):
    c = _client({}, monkeypatch)
    img = Image.new("RGB", (64, 64), "blue")
    r = c.post(
        "/open-vocab/detect",
        json={"model_id": "grounding-dino-tiny", "image_b64": _b64(img), "prompts": ["cat"]},
    )
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "MODEL_MISSING"
    assert "pull" in body["error"]["hint"].lower()


def test_public_mode_autopull_requires_auth(monkeypatch):
    """When public_mode=true and auto_pull_require_auth=true, missing auth blocks auto-pull."""
    c = _client(
        {
            "VISIONSERVEX_SERVER__PUBLIC_MODE": "true",
            "VISIONSERVEX_MODELS__AUTO_PULL": "true",
            "VISIONSERVEX_MODELS__AUTO_PULL_POLICY": "all_auto_downloadable",
            "VISIONSERVEX_MODELS__AUTO_PULL_REQUIRE_AUTH": "true",
        },
        monkeypatch,
    )
    img = Image.new("RGB", (64, 64), "red")
    r = c.post(
        "/open-vocab/detect",
        json={"model_id": "grounding-dino-tiny", "image_b64": _b64(img), "prompts": ["cat"]},
    )
    # auth_enabled=false so policy blocks auto-pull; expect MODEL_MISSING.
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "MODEL_MISSING"


def test_manual_pull_endpoint_returns_clear_error_for_external_models(monkeypatch):
    c = _client({}, monkeypatch)
    r = c.post("/models/grounding-dino-1.5/pull")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "DOWNLOAD_FAILED"


def test_manual_pull_endpoint_returns_clear_error_for_manual_models(monkeypatch):
    c = _client({}, monkeypatch)
    r = c.post("/models/rtmpose-s/pull")
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "MANUAL_DOWNLOAD_REQUIRED"


def test_jobs_endpoint_404_for_missing(monkeypatch):
    c = _client({}, monkeypatch)
    r = c.get("/jobs/no-such-job")
    assert r.status_code == 404


def test_pull_endpoint_synchronous_synthetic_model(monkeypatch):
    c = _client({}, monkeypatch)
    r = c.post("/models/mock-detect/pull?wait=true")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["model_id"] == "mock-detect"


def test_predict_returns_enriched_envelope(monkeypatch):
    c = _client({}, monkeypatch)
    img = Image.new("RGB", (96, 72), "green")
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    buf.seek(0)
    r = c.post(
        "/detect",
        files={"image": ("x.jpg", buf, "image/jpeg")},
        data={"model_id": "mock-detect"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["backend"]
    assert body["precision"]
    assert "model_loaded_from" in body
