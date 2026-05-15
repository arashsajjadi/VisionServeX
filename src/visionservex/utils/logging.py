"""Logging helpers with secret redaction."""

from __future__ import annotations

import logging
import re
from typing import Any

_REDACTED = "***REDACTED***"

# Match common secret-like substrings.
_SECRET_PATTERNS = [
    re.compile(r"(Authorization\s*:\s*Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"(api[_-]?key\s*[:=]\s*)['\"]?[A-Za-z0-9._\-]{8,}['\"]?", re.IGNORECASE),
    re.compile(r"(CF-Access-Client-(?:Id|Secret)\s*:\s*)\S+", re.IGNORECASE),
    re.compile(r"(?P<k>token\s*[:=]\s*)['\"]?[A-Za-z0-9._\-]{8,}['\"]?", re.IGNORECASE),
    # HF token: hf_ prefix followed by alphanumeric chars
    re.compile(r"(HF_TOKEN\s*[:=]\s*)[A-Za-z0-9_\-]{8,}", re.IGNORECASE),
    re.compile(r"(hf_)[A-Za-z0-9]{16,}", re.IGNORECASE),
    # Cloudflare service token secrets
    re.compile(r"(CF-Access-Client-Secret\s*:\s*)\S+", re.IGNORECASE),
    # Base64 image data in logs (long base64 strings prefixed by known patterns)
    re.compile(r"(image_b64\s*[:=]\s*)[A-Za-z0-9+/=]{8,}", re.IGNORECASE),
    re.compile(r"(\"image_b64\"\s*:\s*\")[A-Za-z0-9+/=]{8,}\"", re.IGNORECASE),
]

# Patterns where the entire match should be replaced (not just the suffix)
_FULL_REPLACE_PATTERNS = [
    # JPEG magic bytes in base64 (/9j/...)
    re.compile(r"/9j/[A-Za-z0-9+/=]{8,}", re.IGNORECASE),
    # PNG magic bytes in base64 (iVBORw0...)
    re.compile(r"iVBORw0[A-Za-z0-9+/=]{8,}", re.IGNORECASE),
]


def redact(text: str) -> str:
    """Redact common secret patterns from a string."""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(lambda m: m.group(1) + _REDACTED, redacted)
    for pattern in _FULL_REPLACE_PATTERNS:
        redacted = pattern.sub(_REDACTED, redacted)
    return redacted


class RedactingFilter(logging.Filter):
    """Logging filter that runs ``redact`` over the rendered message."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact(str(record.getMessage()))
            record.args = ()
        except Exception:  # pragma: no cover - defensive
            return True
        return True


def configure_logging(level: str = "INFO") -> None:
    """Idempotently configure root logging with a redacting filter."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    if any(getattr(h, "_visionservex_configured", False) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s :: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    handler.addFilter(RedactingFilter())
    handler._visionservex_configured = True  # type: ignore[attr-defined]
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)


def log_safe_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``data`` with obvious secret keys redacted."""
    sensitive = {"api_key", "token", "authorization", "secret", "password"}
    safe: dict[str, Any] = {}
    for k, v in data.items():
        if k.lower() in sensitive:
            safe[k] = _REDACTED
        elif isinstance(v, dict):
            safe[k] = log_safe_dict(v)
        else:
            safe[k] = v
    return safe


__all__ = ["RedactingFilter", "configure_logging", "get_logger", "log_safe_dict", "redact"]
