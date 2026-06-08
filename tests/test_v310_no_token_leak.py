# SPDX-License-Identifier: Apache-2.0
"""v3.10.0: No HF token leak in tracked files or outputs."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
# Real HF tokens are hf_ followed by ~34 alphanumeric chars (total ~37 chars).
# Use a minimum of 30 chars after hf_ to avoid false positives on field names
# like hf_acceptance_instructions or hf_model_access_status.
_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{30,}")
_FAKE_TOKEN_RE = re.compile(r"hf_ABC|hf_FAKE|hf_TEST|hf_REDACTED", re.IGNORECASE)


def _get_tracked_text_files():
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=30,
        )
        files = result.stdout.splitlines()
        return [
            ROOT / f
            for f in files
            if (ROOT / f).suffix
            in (".py", ".md", ".toml", ".yml", ".yaml", ".json", ".txt", ".cfg", ".ini")
        ]
    except Exception:
        return []


def test_no_real_hf_token_in_tracked_files():
    for fpath in _get_tracked_text_files():
        try:
            text = fpath.read_text(errors="replace")
        except Exception:
            continue
        matches = _TOKEN_RE.findall(text)
        real_matches = [m for m in matches if not _FAKE_TOKEN_RE.match(m)]
        assert not real_matches, (
            f"Possible real HF token in tracked file {fpath.relative_to(ROOT)}: {real_matches}"
        )


def test_byot_runtime_no_token_logging():
    """byot_runtime must not print/log full HF tokens."""
    source = (ROOT / "src/visionservex/byot_runtime.py").read_text()
    assert "print(token" not in source
    assert "log(token" not in source


def test_hf_auth_redacts_token():
    """hf_auth._redact must hide most of a real-looking token."""
    from visionservex.cli.sam3_commands import _redact

    fake = "hf_ABCDEFGHIJ1234567890"
    redacted = _redact(fake)
    assert "ABCDEFGHIJ1234567890" not in redacted
    assert "hf_" in redacted
    assert "***" in redacted
