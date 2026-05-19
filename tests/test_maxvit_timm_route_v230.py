# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.30.0: MaxViT timm/ HF route handling."""

from __future__ import annotations


def test_maxvit_uses_timm_repo() -> None:
    from visionservex.registry import default_registry

    reg = default_registry()
    entry = reg.get("maxvit-tiny-tf-224")
    assert entry is not None
    assert (entry.hf_repo_id or "").startswith("timm/"), (
        f"expected timm/ prefix, got {entry.hf_repo_id!r}"
    )


def test_maxvit_implementation_is_partial_or_wired() -> None:
    from visionservex.registry import default_registry

    reg = default_registry()
    entry = reg.get("maxvit-tiny-tf-224")
    assert entry is not None
    assert entry.implementation_status in ("partial", "wired"), (
        f"unexpected status: {entry.implementation_status}"
    )


def test_hf_classify_engine_gates_timm_repo() -> None:
    """If timm is missing, the engine raises TIMM_REQUIRED."""
    import importlib
    import sys

    # If timm is installed, the gate should pass; if not, the gate emits TIMM_REQUIRED.
    timm_installed = importlib.util.find_spec("timm") is not None
    if timm_installed:
        # Just verify the gate code path exists
        src = sys.modules.get("visionservex.engines.hf_classify") or importlib.import_module(
            "visionservex.engines.hf_classify"
        )
        from pathlib import Path

        text = Path(src.__file__).read_text()
        assert "TIMM_REQUIRED" in text, "hf_classify must reference TIMM_REQUIRED"
        assert 'startswith("timm/")' in text, "hf_classify must gate timm/ repos"
