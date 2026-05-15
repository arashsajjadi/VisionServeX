"""HTTP authentication tests."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from visionservex.config import reload_settings
from visionservex.server.app import create_app


@pytest.fixture
def auth_client(monkeypatch):
    monkeypatch.setenv("VISIONSERVEX_AUTH__ENABLED", "true")
    monkeypatch.setenv("VISIONSERVEX_AUTH__API_KEY", "secret-key-1234567890")
    s = reload_settings()
    app = create_app(s)
    return TestClient(app), s


def test_health_does_not_require_auth():
    app = create_app(reload_settings())
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200


def test_predict_rejects_without_key(auth_client, jpeg_bytes):
    client, _ = auth_client
    r = client.post(
        "/detect",
        files={"image": ("x.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
        data={"model_id": "mock-detect"},
    )
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "UNAUTHENTICATED"


def test_predict_accepts_bearer(auth_client, jpeg_bytes):
    client, _ = auth_client
    r = client.post(
        "/detect",
        files={"image": ("x.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
        data={"model_id": "mock-detect"},
        headers={"Authorization": "Bearer secret-key-1234567890"},
    )
    assert r.status_code == 200


def test_predict_accepts_x_api_key(auth_client, jpeg_bytes):
    client, _ = auth_client
    r = client.post(
        "/detect",
        files={"image": ("x.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
        data={"model_id": "mock-detect"},
        headers={"X-API-Key": "secret-key-1234567890"},
    )
    assert r.status_code == 200


def test_predict_rejects_bad_key(auth_client, jpeg_bytes):
    client, _ = auth_client
    r = client.post(
        "/detect",
        files={"image": ("x.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
        data={"model_id": "mock-detect"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert r.status_code == 401


def test_request_too_large(monkeypatch, jpeg_bytes):
    monkeypatch.setenv("VISIONSERVEX_LIMITS__MAX_UPLOAD_BYTES", "100")
    app = create_app(reload_settings())
    c = TestClient(app)
    r = c.post(
        "/detect",
        files={"image": ("x.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
        data={"model_id": "mock-detect"},
    )
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_task_mismatch_returns_error(jpeg_bytes):
    app = create_app(reload_settings())
    c = TestClient(app)
    r = c.post(
        "/detect",
        files={"image": ("x.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
        data={"model_id": "mock-classify"},  # wrong task
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "TASK_MISMATCH"


def test_url_inputs_disabled_by_default():
    app = create_app(reload_settings())
    c = TestClient(app)
    r = c.post(
        "/open-vocab/detect",
        json={"image_url": "https://example.com/x.jpg", "model_id": "mock-open-vocab"},
    )
    assert r.status_code == 403


def test_base64_input_works():
    import base64

    img = Image.new("RGB", (64, 48), "blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    app = create_app(reload_settings())
    c = TestClient(app)
    r = c.post(
        "/open-vocab/detect",
        json={"image_b64": b64, "model_id": "mock-open-vocab", "prompts": ["cat"]},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["task"] == "open_vocab_detect"
