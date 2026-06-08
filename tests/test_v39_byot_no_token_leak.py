# SPDX-License-Identifier: Apache-2.0
"""v3.9 — Security: no raw HF token in byot_runtime output, source files, or test output."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{30,}")


def test_byot_runtime_source_has_no_hardcoded_token():
    src = Path("src/visionservex/byot_runtime.py").read_text()
    tokens = _TOKEN_RE.findall(src)
    # Only the pattern string itself is allowed as a literal
    real = [t for t in tokens if t not in {"hf_get_token", "hf_redact_token", "hf_auth"}]
    assert not real, f"Hardcoded token-shaped strings in byot_runtime.py: {real}"


def test_hf_auth_source_has_no_hardcoded_token():
    src = Path("src/visionservex/hf_auth.py").read_text()
    # Exclude known function/variable names that happen to start with hf_
    tokens = _TOKEN_RE.findall(src)
    names = {"hf_get_token", "hf_redact_token", "hf_model_access_status",
              "hf_require_user_accepted_license", "hf_hub_download"}
    real = [t for t in tokens if t not in names]
    assert not real, f"Token-shaped strings in hf_auth.py: {real}"


def test_v39_reports_have_no_raw_token():
    reports_dir = Path("notebook/99_final_report/reports")
    if not reports_dir.exists():
        pytest.skip("reports dir not yet created")
    for f in reports_dir.glob("v39_*"):
        text = f.read_text(errors="replace")
        matches = _TOKEN_RE.findall(text)
        real = [m for m in matches if not m.startswith("hf_***")]
        assert not real, f"Raw token-like string in {f.name}: {real}"


def test_v39_artifacts_metadata_have_no_raw_token():
    arts_dir = Path("notebook/99_final_report/artifacts/v39")
    if not arts_dir.exists():
        pytest.skip("v39 artifacts not yet created")
    for f in arts_dir.rglob("*.json"):
        text = f.read_text(errors="replace")
        matches = _TOKEN_RE.findall(text)
        real = [m for m in matches if not m.startswith("hf_***")]
        assert not real, f"Raw token in artifact {f}: {real}"


def test_redacted_token_format():
    from visionservex.hf_auth import hf_redact_token

    fake = "hf_ABCDEFGHIJ1234567890abcdefghij"
    redacted = hf_redact_token(fake)
    assert "ABCDEFGHIJ1234567890abcdefghij" not in redacted
    assert "***" in redacted
    assert len(redacted) < len(fake)
