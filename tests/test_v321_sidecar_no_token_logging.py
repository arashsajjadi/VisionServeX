# SPDX-License-Identifier: Apache-2.0
"""v3.21: sidecar layer never surfaces a raw token. Weight-free."""

from __future__ import annotations

from visionservex.sidecars.errors import SidecarError, SidecarUnavailable, redact


def test_redact_hides_hf_tokens():
    raw = "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
    out = redact(f"connection failed using {raw} at host")
    assert raw not in out
    assert "hf_***REDACTED" in out


def test_redact_handles_none_and_empty():
    assert redact(None) == ""
    assert redact("") == ""


def test_sidecar_error_message_is_redacted():
    raw = "hf_SECRETTOKEN1234567890"
    e = SidecarUnavailable(f"sidecar refused token {raw}")
    assert raw not in str(e)
    assert raw not in e.message


def test_error_repr_does_not_leak_token():
    raw = "hf_ANOTHERSECRET0987654321"
    e = SidecarError(f"boom {raw}")
    assert raw not in repr(e)
