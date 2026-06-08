# SPDX-License-Identifier: Apache-2.0
"""v3.8 — the HF auth layer never leaks a raw token in user-facing payloads."""

from __future__ import annotations

import json

from visionservex import hf_auth as H

FAKE = "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def test_redact_shows_only_5_real_chars():
    r = H.hf_redact_token(FAKE)
    assert r == f"{FAKE[:3]}***{FAKE[-2:]}"
    assert FAKE not in r
    # at most first-3 + last-2 of the real token appear
    assert r.count("*") >= 3


def test_redact_empty_and_short():
    assert H.hf_redact_token("") == ""
    assert H.hf_redact_token(None) == ""
    assert H.hf_redact_token("hf_x") == "***"


def test_whoami_payload_never_contains_raw_token(monkeypatch):
    import pytest

    huggingface_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(H, "_detect", lambda: (FAKE, "env:HF_TOKEN"))

    class _Api:
        def __init__(self, *a, **k):
            pass

        def whoami(self):
            return {
                "name": "tester",
                "type": "user",
                "auth": {"accessToken": {"displayName": "t", "role": "read"}},
                "orgs": [],
            }

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    payload = H.hf_whoami()
    blob = json.dumps(payload, ensure_ascii=False)
    assert FAKE not in blob
    assert payload["token_redacted"] == H.hf_redact_token(FAKE)
    assert payload["name"] == "tester"


def test_validate_payload_never_contains_raw_token(monkeypatch):
    import pytest

    huggingface_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(H, "_detect", lambda: (FAKE, "cli_cache"))

    class _Api:
        def __init__(self, *a, **k):
            pass

        def whoami(self):
            return {"name": "tester", "type": "user", "auth": {"accessToken": {}}}

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    out = H.hf_validate_token()
    assert FAKE not in json.dumps(out)
    assert out["valid"] is True


def test_get_token_redacted_never_returns_raw(monkeypatch):
    monkeypatch.setattr(H, "_detect", lambda: (FAKE, "cli_cache"))
    assert H.hf_get_token(redact=True) == H.hf_redact_token(FAKE)
    # raw only when explicitly requested
    assert H.hf_get_token(redact=False) == FAKE


def test_no_hf_token_string_in_module_source():
    """The auth module must not embed any literal hf_ token."""
    import re
    from pathlib import Path

    src = Path(H.__file__).read_text()
    # any hf_ followed by 20+ token-ish chars would be a leaked token
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", src)
