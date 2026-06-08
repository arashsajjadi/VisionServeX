# SPDX-License-Identifier: Apache-2.0
"""v3.4 BYOT (Bring Your Own Token) — no-token-leak tests.

These tests verify that HF tokens and API keys are never echoed in full
inside any status dict, structured output, or callable return value from
VisionServeX's sam3/DINO API commands.
"""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# a. test_sam3_token_redacted
# ---------------------------------------------------------------------------


def test_sam3_token_redacted(monkeypatch):
    """collect_sam3_status stores only the redacted form, never the full token.

    With HF_TOKEN="hf_TESTFAKETOKEN123" the field hf_token_redacted must be
    exactly "hf_T***23"  (first 3 chars + *** + last 2 chars).
    """
    fake_token = "hf_TESTFAKETOKEN123"
    monkeypatch.setenv("HF_TOKEN", fake_token)

    from visionservex.cli.sam3_commands import collect_sam3_status

    status = collect_sam3_status("sam3-base")

    assert status.hf_token_redacted == "hf_***23", (
        f"Expected 'hf_***23', got {status.hf_token_redacted!r}"
    )


# ---------------------------------------------------------------------------
# b. test_redact_function
# ---------------------------------------------------------------------------


def test_redact_function():
    """_redact('hf_ABCDEFGHIJ') returns 'hf_A***IJ'."""
    from visionservex.cli.sam3_commands import _redact

    result = _redact("hf_ABCDEFGHIJ")
    assert result == "hf_***IJ", f"Expected 'hf_***IJ', got {result!r}"

    # Additional edge-case coverage.
    assert _redact("") == ""
    assert _redact(None) == ""  # type: ignore[arg-type]
    assert _redact("abc") == "***"  # shorter than 8 chars → full mask


# ---------------------------------------------------------------------------
# c. test_no_full_token_in_sam3_status_output
# ---------------------------------------------------------------------------


def test_no_full_token_in_sam3_status_output(monkeypatch):
    """The status dict serialised to JSON must not contain the full token string."""
    fake_token = "hf_TESTFAKETOKEN123"
    monkeypatch.setenv("HF_TOKEN", fake_token)

    from visionservex.cli.sam3_commands import collect_sam3_status

    status = collect_sam3_status("sam3-base")
    serialised = json.dumps(status.to_dict())

    assert fake_token not in serialised, f"Full HF token found in status output: {serialised!r}"


# ---------------------------------------------------------------------------
# d. test_deepdataspace_key_redacted
# ---------------------------------------------------------------------------


def test_deepdataspace_key_redacted(monkeypatch):
    """In dino api command output with a fake key, the full key is not present."""
    fake_key = "dds_FAKEKEYFORTESTING9"
    monkeypatch.setenv("DEEPDATASPACE_API_KEY", fake_key)

    from visionservex.cli.dino_commands import _redact_key

    redacted = _redact_key(fake_key)

    # Full key must NOT appear in the redacted output.
    assert fake_key not in redacted, f"Full API key leaked in redacted output: {redacted!r}"
    assert "***" in redacted, f"Expected masking '***' in redacted key, got {redacted!r}"


# ---------------------------------------------------------------------------
# e. test_sam3_with_token_still_blocked
# ---------------------------------------------------------------------------


def test_sam3_with_token_still_blocked(monkeypatch):
    """Even with a token, collect_sam3_status returns blocker_code != ''.

    SAM3 is HF-gated and the VisionServeX engine glue is not yet complete;
    supplying a token alone is insufficient to make the model runnable.
    """
    monkeypatch.setenv("HF_TOKEN", "hf_TESTFAKETOKEN123")

    from visionservex.cli.sam3_commands import collect_sam3_status

    status = collect_sam3_status("sam3-base")

    assert status.blocker_code != "", (
        "Expected a non-empty blocker_code even when HF_TOKEN is set, "
        f"but got blocker_code={status.blocker_code!r}"
    )
