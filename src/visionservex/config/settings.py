"""Application configuration.

Settings load from (in priority order):
  1. Explicit overrides passed to ``Settings(...)`` or ``reload_settings(**)``.
  2. Environment variables prefixed with ``VISIONSERVEX_`` (nested via ``__``).
  3. A YAML file pointed to by ``VISIONSERVEX_CONFIG_FILE``.
  4. Built-in defaults.

Defaults are designed for safe local use.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from platformdirs import user_cache_dir
from pydantic import BaseModel, Field, IPvAnyAddress, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_cache_dir() -> Path:
    override = os.environ.get("VISION_SERVEX_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(user_cache_dir("visionservex", appauthor=False))


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    public_mode: bool = False
    workers: int = Field(default=1, ge=1, le=64)

    @field_validator("host")
    @classmethod
    def _validate_host(cls, value: str) -> str:
        # Accept named hosts but warn-by-validator that 0.0.0.0 is intentional.
        if value in {"", None}:
            return "127.0.0.1"
        # We do not block 0.0.0.0 here because power-users may want it inside a
        # firewalled container, but the server module emits a loud banner.
        return value


class AuthConfig(BaseModel):
    enabled: bool = False
    api_key: str | None = None
    header_name: str = "Authorization"
    accept_cf_access_headers: bool = True


class LimitsConfig(BaseModel):
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    max_image_pixels: int = Field(default=33_177_600, ge=1)  # ~7680x4320
    max_image_dim: int = Field(default=8192, ge=64)
    rate_limit_per_minute: int = Field(default=120, ge=0)
    rate_limit_burst: int = Field(default=20, ge=0)


class CorsConfig(BaseModel):
    allowed_origins: list[str] = Field(default_factory=list)
    allow_credentials: bool = False
    allow_methods: list[str] = Field(default_factory=lambda: ["GET", "POST"])
    allow_headers: list[str] = Field(
        default_factory=lambda: ["Authorization", "Content-Type"]
    )


class CacheConfig(BaseModel):
    cache_dir: Path = Field(default_factory=_default_cache_dir)
    offline: bool = False
    download_chunk_bytes: int = Field(default=1024 * 1024, ge=4096)


class RuntimeConfig(BaseModel):
    device_preference: Literal["auto", "cuda", "mps", "cpu", "rocm", "directml"] = "auto"
    precision_preference: Literal["auto", "fp32", "fp16", "bf16", "int8"] = "auto"
    max_loaded_models: int = Field(default=2, ge=1, le=64)
    per_model_concurrency: int = Field(default=2, ge=1, le=64)
    queue_size: int = Field(default=64, ge=1)
    request_timeout_s: float = Field(default=60.0, gt=0)
    model_idle_unload_s: float = Field(default=600.0, ge=0)


class InputsConfig(BaseModel):
    allow_url_inputs: bool = False
    allow_local_paths: bool = False
    allowed_mime_types: list[str] = Field(
        default_factory=lambda: [
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/bmp",
            "image/tiff",
        ]
    )
    url_allowlist_hosts: list[str] = Field(default_factory=list)
    url_request_timeout_s: float = Field(default=10.0, gt=0)
    url_max_bytes: int = Field(default=20 * 1024 * 1024, ge=1)


class ModelsConfig(BaseModel):
    """Behavior around model availability and downloads."""

    auto_pull: bool = False
    auto_pull_policy: Literal[
        "never", "easy_only", "registry_allowed", "all_auto_downloadable"
    ] = "easy_only"
    auto_pull_timeout_s: float = 600.0
    auto_pull_max_size_gb: float = 5.0
    auto_pull_require_auth: bool = True
    allow_mock_fallback: bool = False  # honest by default: never fake real engines


class TunnelConfig(BaseModel):
    provider: Literal["cloudflare", "none"] = "cloudflare"
    tunnel_name: str = "visionservex"
    config_dir: Path = Field(
        default_factory=lambda: Path.home() / ".visionservex" / "tunnel"
    )
    require_auth_for_public: bool = True


def _yaml_config_source(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config at {file_path} must be a mapping")
    return data


class Settings(BaseSettings):
    """Top-level settings object."""

    model_config = SettingsConfigDict(
        env_prefix="VISIONSERVEX_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    server: ServerConfig = Field(default_factory=ServerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    cors: CorsConfig = Field(default_factory=CorsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    inputs: InputsConfig = Field(default_factory=InputsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    tunnel: TunnelConfig = Field(default_factory=TunnelConfig)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @classmethod
    def load(cls, **overrides: Any) -> Settings:
        """Build a Settings instance, layering YAML on top of env defaults."""
        yaml_path_str = os.environ.get("VISIONSERVEX_CONFIG_FILE")
        yaml_overrides: dict[str, Any] = {}
        if yaml_path_str:
            yaml_overrides = _yaml_config_source(Path(yaml_path_str).expanduser())
        # Pydantic-settings will fill from env; we then merge YAML and explicit overrides.
        base = cls()
        merged = base.model_dump()
        _deep_update(merged, yaml_overrides)
        _deep_update(merged, overrides)
        return cls.model_validate(merged)

    def public_safety_warnings(self) -> list[str]:
        """Return strings describing dangerous public configurations."""
        warnings: list[str] = []
        if self.server.public_mode and not self.auth.enabled:
            warnings.append(
                "PUBLIC MODE IS ENABLED BUT AUTHENTICATION IS DISABLED. "
                "Anyone able to reach the server can run inference."
            )
        if self.auth.enabled and not self.auth.api_key:
            warnings.append(
                "AUTH IS ENABLED BUT API_KEY IS EMPTY. All requests will be rejected."
            )
        if self.server.host == "0.0.0.0" and not self.auth.enabled:
            warnings.append(
                "SERVER IS BOUND TO 0.0.0.0 WITHOUT AUTHENTICATION. "
                "This will expose the API to the local network."
            )
        if self.inputs.allow_url_inputs and not self.inputs.url_allowlist_hosts:
            warnings.append(
                "Remote URL inputs are enabled with no host allowlist. "
                "SSRF protection is on, but consider setting an allowlist."
            )
        return warnings


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load()


def reload_settings(**overrides: Any) -> Settings:
    """Force-reload settings (used by tests and the CLI)."""
    get_settings.cache_clear()
    settings = Settings.load(**overrides)
    # Re-seed the cache with the new instance.
    get_settings.cache_clear()
    return settings


__all__ = [
    "Settings",
    "ServerConfig",
    "AuthConfig",
    "LimitsConfig",
    "CorsConfig",
    "CacheConfig",
    "RuntimeConfig",
    "InputsConfig",
    "ModelsConfig",
    "TunnelConfig",
    "get_settings",
    "reload_settings",
]
