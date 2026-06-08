# SPDX-License-Identifier: Apache-2.0
"""v3.8 — `visionservex hf` CLI: status/whoami/connect/logout/check-model."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from visionservex import hf_auth as H
from visionservex.cli.main import app

runner = CliRunner()
FAKE = "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def test_hf_group_registered():
    res = runner.invoke(app, ["hf", "--help"])
    assert res.exit_code == 0
    for cmd in ("status", "whoami", "connect", "logout", "check-model"):
        assert cmd in res.output


def test_hf_status_json_no_token(monkeypatch):
    monkeypatch.setattr(H, "_detect", lambda: (None, None))
    res = runner.invoke(app, ["hf", "status", "--json"])
    assert res.exit_code == 0
    payload = json.loads(res.output)
    assert payload["logged_in"] is False
    assert payload["token_source"] is None
    assert payload["token_redacted"] == ""


def test_hf_status_json_with_token_redacted(monkeypatch):
    monkeypatch.setattr(H, "_detect", lambda: (FAKE, "cli_cache"))
    monkeypatch.setattr(H, "hf_whoami", lambda redact=True: {
        "logged_in": True, "name": "tester", "type": "user",
        "token_display_name": "tok", "token_role": "read", "orgs": [],
        "token_redacted": H.hf_redact_token(FAKE), "error": None})
    res = runner.invoke(app, ["hf", "status", "--json"])
    assert res.exit_code == 0
    assert FAKE not in res.output
    payload = json.loads(res.output)
    assert payload["logged_in"] is True
    assert payload["token_redacted"] == H.hf_redact_token(FAKE)


def test_hf_connect_no_token_in_env(monkeypatch):
    monkeypatch.delenv("MISSING_TOKEN_VAR", raising=False)
    res = runner.invoke(app, ["hf", "connect", "--token-env", "MISSING_TOKEN_VAR", "--json"])
    assert res.exit_code == 1
    assert "NO_TOKEN_IN_ENV" in res.output


def test_hf_connect_invalid_token_file(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("not-a-token")
    res = runner.invoke(app, ["hf", "connect", "--token-file", str(f), "--json"])
    assert res.exit_code == 1
    assert "TOKEN_FILE_INVALID" in res.output


def test_hf_whoami_not_logged_in(monkeypatch):
    monkeypatch.setattr(H, "hf_whoami", lambda redact=True: {"logged_in": False})
    res = runner.invoke(app, ["hf", "whoami"])
    assert res.exit_code == 1
    assert "Not logged in" in res.output
