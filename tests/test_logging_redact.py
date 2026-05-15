"""Log redaction tests."""

from __future__ import annotations

from visionservex.utils.logging import log_safe_dict, redact


def test_redact_bearer():
    out = redact("Authorization: Bearer abcdef123456")
    assert "abcdef123456" not in out
    assert "REDACTED" in out


def test_redact_api_key():
    out = redact("api_key=supersecretvalue")
    assert "supersecretvalue" not in out


def test_redact_cf_access_secret():
    out = redact("CF-Access-Client-Secret: VeryLongSecret123")
    assert "VeryLongSecret123" not in out


def test_log_safe_dict_nested():
    d = {"a": 1, "auth": {"api_key": "topsecret"}}
    safe = log_safe_dict(d)
    assert safe["auth"]["api_key"] == "***REDACTED***"
    assert safe["a"] == 1
