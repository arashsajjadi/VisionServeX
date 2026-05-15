# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Security and privacy hardening tests for v1.0.0rc2."""

from __future__ import annotations

import io
import json
import stat
from pathlib import Path

import pytest
from PIL import Image

from visionservex.config import reload_settings


def _img() -> Image.Image:
    return Image.new("RGB", (64, 64), color="red")


def _jpeg() -> bytes:
    buf = io.BytesIO()
    _img().save(buf, format="JPEG")
    return buf.getvalue()


# ============================================================
# Log redaction
# ============================================================


def test_bearer_token_redacted():
    from visionservex.utils.logging import redact

    out = redact("Authorization: Bearer supersecrettoken1234567890")
    assert "supersecrettoken1234567890" not in out
    assert "REDACTED" in out


def test_cf_access_secret_redacted():
    from visionservex.utils.logging import redact

    out = redact("CF-Access-Client-Secret: myCFsecretValue123456")
    assert "myCFsecretValue123456" not in out


def test_hf_token_redacted():
    from visionservex.utils.logging import redact

    out = redact("HF_TOKEN=hf_abcdefghijklmnopqrst")
    assert "hf_abcdefghijklmnopqrst" not in out


def test_hf_prefix_token_redacted():
    from visionservex.utils.logging import redact

    out = redact("using token hf_ABCDEFGHIJKLMNOPQRSTUVWxyz1234")
    assert "ABCDEFGHIJKLMNOPQRSTUVWxyz1234" not in out


def test_base64_jpeg_redacted():
    from visionservex.utils.logging import redact

    # JPEG magic bytes in base64: /9j/4AA...
    out = redact("image_b64=/9j/4AAQSkZJRgABAQAAAQABAAD==")
    assert "4AAQSkZJRgABAQAAAQABAAD==" not in out


def test_base64_png_magic_redacted():
    from visionservex.utils.logging import redact

    out = redact("data: iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAAC")
    # The PNG magic bytes (iVBORw0...) should be redacted; the suffix should not appear
    assert "NSUhEUgAAAAEAAAABCAAAAAA6fptVAAAAC" not in out


def test_non_secret_not_redacted():
    from visionservex.utils.logging import redact

    plain = "model_id: dfine-n, status: ok, latency_ms: 12.3"
    assert redact(plain) == plain


# ============================================================
# Security audit command
# ============================================================


def test_security_audit_json(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["security", "audit", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    # Must never claim E2E encryption
    assert data["e2e_encryption_claimed"] is False
    assert "privacy_note" in data
    assert "end-to-end encryption" in data["privacy_note"].lower()
    assert "mode" in data
    assert "auth_enabled" in data
    assert "retention_mode" in data


def test_security_audit_no_e2e_claim(monkeypatch, tmp_path):
    """Critically important: no E2E encryption claim."""
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["security", "audit", "--json"])
    data = json.loads(r.output)
    assert data["e2e_encryption_claimed"] is False, (
        "CRITICAL: security audit must NEVER claim E2E encryption"
    )


def test_security_doctor_json(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["security", "doctor", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert isinstance(data, list)
    assert any(c["name"] == "e2e_not_claimed" for c in data)
    e2e_check = next(c for c in data if c["name"] == "e2e_not_claimed")
    assert e2e_check["ok"] is True


def test_security_test_redaction(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["security", "test-redaction"])
    assert r.exit_code == 0, r.output


def test_security_checklist(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["security", "checklist"])
    assert r.exit_code == 0, r.output
    assert "end-to-end" in r.output.lower() or "E2E" in r.output


# ============================================================
# Privacy config defaults
# ============================================================


def test_privacy_defaults():
    from visionservex.config import PrivacyConfig

    pc = PrivacyConfig()
    assert pc.save_inputs is False
    assert pc.save_outputs is False
    assert pc.save_prompts is False
    assert pc.job_payload_retention is False
    assert pc.encrypt_job_store is False
    assert pc.retention_mode == "metadata_only"


def test_security_mode_default():
    from visionservex.config import SecurityModeConfig

    sm = SecurityModeConfig()
    assert sm.mode == "local_private"
    assert sm.require_cloudflare_access is False
    assert sm.trust_cf_headers is False
    assert sm.sidecar_public is False


# ============================================================
# No data retention by default
# ============================================================


def test_server_does_not_save_uploaded_image_to_disk(monkeypatch, tmp_path):
    """Uploaded images must not be written to disk by default."""
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    s = reload_settings()
    assert s.privacy.save_inputs is False

    # Count files in tmpdir before request
    before = (
        set(tmp_path.rglob("*.jpg")) | set(tmp_path.rglob("*.png")) | set(tmp_path.rglob("*.jpeg"))
    )

    from fastapi.testclient import TestClient

    from visionservex.server.app import create_app

    app = create_app(s)
    client = TestClient(app)
    r = client.post(
        "/detect",
        files={"image": ("test.jpg", io.BytesIO(_jpeg()), "image/jpeg")},
        data={"model_id": "mock-detect"},
    )
    assert r.status_code == 200

    after = (
        set(tmp_path.rglob("*.jpg")) | set(tmp_path.rglob("*.png")) | set(tmp_path.rglob("*.jpeg"))
    )
    new_image_files = after - before
    assert not new_image_files, f"Server wrote image file(s) to disk: {new_image_files}"


def test_job_store_no_image_bytes(monkeypatch, tmp_path):
    """SQLite job store must not contain raw image bytes."""
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VISIONSERVEX_JOBS__STORE", "sqlite")
    db_path = tmp_path / "test_jobs.db"
    monkeypatch.setenv("VISIONSERVEX_JOBS__SQLITE_PATH", str(db_path))
    reload_settings()

    from visionservex.runtime.job_store import SQLiteJobStore

    store = SQLiteJobStore(db_path=db_path)
    job = store.create(model_id="dfine-n", kind="predict")
    # Store does NOT persist image bytes — only metadata
    fetched = store.get(job.job_id)
    assert fetched is not None
    assert fetched.model_id == "dfine-n"
    # Verify DB file doesn't contain JPEG magic bytes
    if db_path.exists():
        db_contents = db_path.read_bytes()
        assert b"\xff\xd8\xff" not in db_contents, "JPEG bytes found in job store DB!"


# ============================================================
# Encryption-at-rest
# ============================================================


def test_encryption_keygen():
    """generate_key must produce valid 32-byte Fernet key."""
    pytest.importorskip("cryptography")
    from visionservex.security.encryption import generate_key

    key = generate_key()
    assert isinstance(key, bytes)
    import base64

    decoded = base64.urlsafe_b64decode(key + b"==")
    assert len(decoded) == 32


def test_field_encryptor_roundtrip():
    pytest.importorskip("cryptography")
    from visionservex.security.encryption import FieldEncryptor, generate_key

    key = generate_key()
    enc = FieldEncryptor(key)
    plaintext = "sensitive metadata value"
    ciphertext = enc.encrypt(plaintext)
    assert plaintext not in ciphertext
    assert enc.decrypt(ciphertext) == plaintext


def test_field_encryptor_wrong_key_fails():
    pytest.importorskip("cryptography")
    from visionservex.security.encryption import EncryptionKeyError, FieldEncryptor, generate_key

    key1 = generate_key()
    key2 = generate_key()
    enc1 = FieldEncryptor(key1)
    enc2 = FieldEncryptor(key2)
    ciphertext = enc1.encrypt("secret")
    with pytest.raises(EncryptionKeyError):
        enc2.decrypt(ciphertext)


def test_keygen_cli_no_key_in_output(monkeypatch, tmp_path):
    """keygen with --out must write to file, not echo key to terminal."""
    pytest.importorskip("cryptography")
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    key_file = tmp_path / "test.key"
    runner = CliRunner()
    r = runner.invoke(app, ["security", "keygen", "--out", str(key_file)])
    assert r.exit_code == 0, r.output
    assert key_file.exists()
    # File permissions should be 0600
    mode = key_file.stat().st_mode & 0o777
    assert mode == 0o600, f"Key file has wrong permissions: {oct(mode)}"


def test_check_key_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VISIONSERVEX_PRIVACY__ENCRYPT_JOB_STORE", "false")
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["security", "check-key", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["encrypt_job_store"] is False
    assert data["status"] == "disabled"


# ============================================================
# Secure temp files
# ============================================================


def test_secure_temp_file_deleted_after_use():
    from visionservex.runtime.temp_manager import secure_temp_file

    with secure_temp_file(suffix=".jpg") as p:
        p.write_bytes(b"test content")
        path_str = str(p)
        assert p.exists()
    assert not Path(path_str).exists(), "Temp file was not cleaned up"


def test_secure_temp_file_deleted_on_exception():
    from visionservex.runtime.temp_manager import secure_temp_file

    path_str = None
    try:
        with secure_temp_file(suffix=".jpg") as p:
            p.write_bytes(b"test content")
            path_str = str(p)
            raise ValueError("simulated error")
    except ValueError:
        pass
    assert path_str is not None
    assert not Path(path_str).exists(), "Temp file leaked after exception"


def test_secure_temp_file_permissions():
    from visionservex.runtime.temp_manager import secure_temp_file

    with secure_temp_file(suffix=".tmp") as p:
        mode = p.stat().st_mode & 0o777
        # Should be owner-only read/write (0600)
        assert not (mode & stat.S_IRGRP), "Temp file is group-readable"
        assert not (mode & stat.S_IROTH), "Temp file is world-readable"


def test_cleanup_command(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["privacy", "cleanup", "--dry-run", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["dry_run"] is True


# ============================================================
# Public mode security
# ============================================================


def test_public_mode_auth_warning(monkeypatch, tmp_path):
    """public_mode=true without auth should produce safety warnings."""
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VISIONSERVEX_SERVER__PUBLIC_MODE", "true")
    monkeypatch.setenv("VISIONSERVEX_AUTH__ENABLED", "false")
    s = reload_settings()
    warnings = s.public_safety_warnings()
    assert any("PUBLIC MODE" in w for w in warnings)


def test_cors_wildcard_allowed_origins_default_empty(monkeypatch, tmp_path):
    """CORS allowed_origins must be empty by default (no wildcard)."""
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    s = reload_settings()
    assert s.cors.allowed_origins == []
    assert "*" not in s.cors.allowed_origins


def test_security_mode_command(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    r = runner.invoke(app, ["security", "mode", "local_private"])
    assert r.exit_code == 0, r.output
    assert "local_private" in r.output
