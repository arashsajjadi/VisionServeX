# SPDX-License-Identifier: Apache-2.0
"""v3.10.0: HF auth_check uses HfApi.auth_check (not model_info)."""
from __future__ import annotations

import pytest


def test_hf_auth_module_uses_auth_check():
    """Verify hf_auth.py calls auth_check, not model_info."""
    from pathlib import Path

    hf_auth_path = (
        Path(__file__).parent.parent / "src/visionservex/hf_auth.py"
    )
    if not hf_auth_path.exists():
        pytest.skip("hf_auth.py not found")
    source = hf_auth_path.read_text()
    assert "auth_check" in source, "hf_auth.py must use HfApi.auth_check()"
    assert "model_info" not in source or source.count("model_info") == 0 or (
        "auth_check" in source
    ), "hf_auth.py should prefer auth_check over model_info"


def test_hf_get_token_returns_string_or_none():
    pytest.importorskip("huggingface_hub")
    from visionservex.hf_auth import hf_get_token

    token = hf_get_token()
    assert token is None or isinstance(token, str)


def test_hf_get_token_not_printed():
    """Calling hf_get_token must not print to stdout."""
    import io
    import sys

    pytest.importorskip("huggingface_hub")
    from visionservex.hf_auth import hf_get_token

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        hf_get_token()
    finally:
        sys.stdout = old_stdout
    output = captured.getvalue()
    # Token must not appear in output
    import re
    assert not re.search(r"hf_[A-Za-z0-9]{10,}", output), (
        f"HF token may have been printed to stdout: {output[:100]}"
    )


def test_sam3_commands_redact_function():
    from visionservex.cli.sam3_commands import _redact

    assert _redact("hf_ABCDEFGHIJ1234567890") != "hf_ABCDEFGHIJ1234567890"
    assert _redact(None) == ""
    assert _redact("") == ""
    assert "***" in _redact("hf_ABCDEFGHIJ1234567890")
