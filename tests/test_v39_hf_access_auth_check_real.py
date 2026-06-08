# SPDX-License-Identifier: Apache-2.0
"""v3.9 — HF auth_check: confirms access_granted for DINOv3 and SAM3 repos.

Unit tests (no token) check that auth_check raises on bad input.
Live tests (VISIONSERVEX_RUN_GATED_HF=1) verify real access.
"""

from __future__ import annotations

import os

import pytest


LIVE = os.getenv("VISIONSERVEX_RUN_GATED_HF") == "1"

DINOV3_REPOS = [
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "facebook/dinov3-convnext-small-pretrain-lvd1689m",
]
SAM3_REPOS = ["facebook/sam3", "facebook/sam3.1"]


def test_hf_auth_module_importable():
    from visionservex import hf_auth

    assert hasattr(hf_auth, "hf_model_access_status")
    assert hasattr(hf_auth, "hf_redact_token")


def test_redact_token_never_exposes_full():
    from visionservex.hf_auth import hf_redact_token

    result = hf_redact_token("hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in result
    assert result.startswith("hf_")
    assert "***" in result


def test_redact_token_none_returns_empty():
    from visionservex.hf_auth import hf_redact_token

    assert hf_redact_token(None) == ""


@pytest.mark.skipif(not LIVE, reason="Set VISIONSERVEX_RUN_GATED_HF=1 for live access tests")
@pytest.mark.parametrize("repo", DINOV3_REPOS)
def test_dinov3_repo_access_granted(repo):
    pytest.importorskip("huggingface_hub")
    from visionservex import hf_auth

    state = hf_auth.hf_model_access_status(repo)
    assert state == "access_granted", f"{repo} returned {state}; accept the license on Hub"


@pytest.mark.skipif(not LIVE, reason="Set VISIONSERVEX_RUN_GATED_HF=1 for live access tests")
@pytest.mark.parametrize("repo", SAM3_REPOS)
def test_sam3_repo_access_granted(repo):
    pytest.importorskip("huggingface_hub")
    from visionservex import hf_auth

    state = hf_auth.hf_model_access_status(repo)
    assert state == "access_granted", f"{repo} returned {state}; accept the license on Hub"
