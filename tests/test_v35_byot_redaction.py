# SPDX-License-Identifier: Apache-2.0
"""v3.5 BYOT token redaction tests (extended)."""

from __future__ import annotations

import json


def test_redact_function_v35():
    from visionservex.cli.sam3_commands import _redact

    assert _redact("hf_ABCDEFGHIJ") == "hf_***IJ"
    assert _redact("") == ""
    assert _redact(None) == ""
    assert _redact("abc") == "***"


def test_sam3_token_redacted_v35(monkeypatch):
    fake_token = "hf_TESTFAKETOKEN123"
    monkeypatch.setenv("HF_TOKEN", fake_token)
    from visionservex.cli.sam3_commands import collect_sam3_status

    status = collect_sam3_status("sam3-base")
    assert status.hf_token_redacted == "hf_***23"
    assert fake_token not in json.dumps(status.to_dict())


def test_dino_api_key_redacted_v35(monkeypatch):
    fake_key = "dds_FAKEKEYFORTESTING9"
    monkeypatch.setenv("DEEPDATASPACE_API_KEY", fake_key)
    from visionservex.cli.dino_commands import _redact_key

    redacted = _redact_key(fake_key)
    assert fake_key not in redacted
    assert "***" in redacted


def test_no_full_token_in_any_status(monkeypatch):
    fake_token = "hf_TESTFAKETOKEN123"
    monkeypatch.setenv("HF_TOKEN", fake_token)
    from visionservex.cli.sam3_commands import collect_sam3_status

    status = collect_sam3_status("sam3-base")
    serialised = json.dumps(status.to_dict())
    assert fake_token not in serialised, f"Full HF token in output: {serialised!r}"


def test_sam3_still_blocked_with_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_TESTFAKETOKEN123")
    from visionservex.cli.sam3_commands import collect_sam3_status

    status = collect_sam3_status("sam3-base")
    assert status.blocker_code != "", "sam3-base must still be blocked even with token"
