"""Security validator tests."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from visionservex.config import InputsConfig, LimitsConfig
from visionservex.security.ssrf import URLValidationError, validate_remote_url
from visionservex.security.validators import (
    InputValidationError,
    validate_image_bytes,
    validate_mime_type,
    validate_path_input,
)
from visionservex.utils.paths import PathTraversalError, safe_join


def test_validate_mime_accepts_jpeg():
    inputs = InputsConfig()
    assert validate_mime_type("image/jpeg", inputs) == "image/jpeg"


def test_validate_mime_rejects_text():
    inputs = InputsConfig()
    with pytest.raises(InputValidationError):
        validate_mime_type("text/plain", inputs)


def test_validate_image_bytes_rejects_too_large(jpeg_bytes):
    limits = LimitsConfig(max_upload_bytes=10)
    with pytest.raises(InputValidationError):
        validate_image_bytes(jpeg_bytes, limits)


def test_validate_image_bytes_accepts_normal(jpeg_bytes):
    limits = LimitsConfig()
    validate_image_bytes(jpeg_bytes, limits)


def test_validate_image_bytes_rejects_dim():
    big = Image.new("RGB", (200, 200), "blue")
    buf = io.BytesIO(); big.save(buf, format="PNG")
    limits = LimitsConfig(max_image_dim=64)
    with pytest.raises(InputValidationError):
        validate_image_bytes(buf.getvalue(), limits)


def test_safe_join_rejects_traversal(tmp_path):
    with pytest.raises(PathTraversalError):
        safe_join(tmp_path, "../etc/passwd")
    with pytest.raises(PathTraversalError):
        safe_join(tmp_path, "/etc/passwd")


def test_safe_join_accepts(tmp_path):
    p = safe_join(tmp_path, "sub/x.txt")
    assert str(p).startswith(str(tmp_path.resolve()))


def test_validate_path_input_disabled_by_default(tmp_path):
    inputs = InputsConfig(allow_local_paths=False)
    with pytest.raises(InputValidationError):
        validate_path_input("x.txt", root=tmp_path, inputs=inputs)


def test_url_validation_requires_https():
    with pytest.raises(URLValidationError):
        validate_remote_url("http://example.com/x.jpg")


def test_url_validation_rejects_localhost():
    with pytest.raises(URLValidationError):
        validate_remote_url("https://localhost/x.jpg")


def test_url_validation_rejects_private_ip():
    with pytest.raises(URLValidationError):
        validate_remote_url("https://127.0.0.1/x.jpg")


def test_url_validation_rejects_file_scheme():
    with pytest.raises(URLValidationError):
        validate_remote_url("file:///etc/passwd")


def test_url_validation_rejects_userinfo():
    with pytest.raises(URLValidationError):
        validate_remote_url("https://user:pw@example.com/x.jpg")


def test_url_validation_allowlist_blocks():
    with pytest.raises(URLValidationError):
        validate_remote_url(
            "https://example.com/x.jpg",
            allowlist_hosts=["images.example.org"],
        )
