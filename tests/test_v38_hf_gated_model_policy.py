# SPDX-License-Identifier: Apache-2.0
"""v3.8 — gated-model access logic uses auth_check and refuses without acceptance."""

from __future__ import annotations

from visionservex import hf_auth as H


def test_acceptance_instructions_for_gated(monkeypatch):
    info = H.hf_acceptance_instructions("sam3-base")
    assert info["known"] is True
    assert info["hf_repo"] == "facebook/sam3"
    assert any("Agree and access repository" in s for s in info["instructions"])
    assert "redistribute" in info["warning"].lower()


def test_access_status_no_token_is_auth_required(monkeypatch):
    monkeypatch.setattr(H, "hf_get_token", lambda redact=False: None)
    monkeypatch.setattr(H, "hf_is_logged_in", lambda: False)
    monkeypatch.setattr(H, "hf_token_source", lambda: None)
    a = H.hf_model_access_status("sam3-base")
    assert a["state"] == "auth_required"
    assert "hf connect" in a["next_command"]


def test_access_status_gated_not_accepted(monkeypatch):
    monkeypatch.setattr(H, "hf_get_token", lambda redact=False: "hf_fake")
    monkeypatch.setattr(H, "hf_is_logged_in", lambda: True)
    monkeypatch.setattr(H, "hf_token_source", lambda: "cli_cache")

    import huggingface_hub
    from huggingface_hub.utils import GatedRepoError

    class _Api:
        def __init__(self, *a, **k):
            pass

        def auth_check(self, repo):
            # construct without the response kwarg required by __init__
            raise GatedRepoError.__new__(GatedRepoError)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    a = H.hf_model_access_status("dinov3-vitb16")
    assert a["state"] == "auth_required_license_pending"
    assert "acceptance" in a


def test_access_status_granted(monkeypatch):
    monkeypatch.setattr(H, "hf_get_token", lambda redact=False: "hf_fake")
    monkeypatch.setattr(H, "hf_is_logged_in", lambda: True)
    monkeypatch.setattr(H, "hf_token_source", lambda: "cli_cache")

    import huggingface_hub

    class _Api:
        def __init__(self, *a, **k):
            pass

        def auth_check(self, repo):
            return None

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    a = H.hf_model_access_status("sam3-base")
    assert a["state"] == "access_granted"


def test_external_api_model_is_not_local_access():
    a = H.hf_model_access_status("dino-x-api")
    assert a["state"] == "external_api_only"
