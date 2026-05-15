"""Cloudflare tunnel config generation."""

from __future__ import annotations

import yaml

from visionservex.tunnel import cloudflared_doctor, generate_config, public_checklist


def test_doctor_returns_dict():
    r = cloudflared_doctor()
    assert "installed" in r
    assert "install_hint" in r


def test_generate_config_has_catch_all_404():
    cfg = generate_config(tunnel_name="vsx", hostname="api.example.com")
    # strip comments before parsing
    payload = "\n".join(line for line in cfg.splitlines() if not line.strip().startswith("#"))
    data = yaml.safe_load(payload)
    assert data["tunnel"] == "vsx"
    ingress = data["ingress"]
    assert ingress[-1] == {"service": "http_status:404"}
    assert any(rule.get("hostname") == "api.example.com" for rule in ingress)


def test_public_checklist_warns_when_auth_off():
    items = public_checklist(auth_enabled=False)
    assert any("AUTH IS CURRENTLY DISABLED" in x for x in items)


def test_public_checklist_no_disabled_when_auth_on():
    items = public_checklist(auth_enabled=True)
    assert not any("CURRENTLY DISABLED" in x for x in items)
