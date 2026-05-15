"""Config loader tests."""

from __future__ import annotations

from visionservex.config import Settings, reload_settings


def test_default_safe_settings():
    s = Settings.load()
    assert s.server.host == "127.0.0.1"
    assert s.server.public_mode is False
    assert s.auth.enabled is False
    assert s.inputs.allow_url_inputs is False
    assert s.inputs.allow_local_paths is False


def test_env_override(monkeypatch):
    monkeypatch.setenv("VISIONSERVEX_SERVER__PORT", "9999")
    monkeypatch.setenv("VISIONSERVEX_AUTH__ENABLED", "true")
    monkeypatch.setenv("VISIONSERVEX_AUTH__API_KEY", "x" * 40)
    s = reload_settings()
    assert s.server.port == 9999
    assert s.auth.enabled is True
    assert s.auth.api_key == "x" * 40


def test_public_safety_warnings_when_unsafe():
    s = Settings.load(server={"public_mode": True}, auth={"enabled": False})
    warnings = s.public_safety_warnings()
    assert any("PUBLIC MODE" in w for w in warnings)


def test_public_safety_no_warnings_when_safe():
    s = Settings.load(
        server={"public_mode": True},
        auth={"enabled": True, "api_key": "x" * 40},
    )
    warnings = s.public_safety_warnings()
    assert not any("PUBLIC MODE IS ENABLED BUT AUTHENTICATION IS DISABLED" in w for w in warnings)


def test_yaml_overrides(monkeypatch, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("server:\n  port: 7777\n", encoding="utf-8")
    monkeypatch.setenv("VISIONSERVEX_CONFIG_FILE", str(cfg))
    s = reload_settings()
    assert s.server.port == 7777
