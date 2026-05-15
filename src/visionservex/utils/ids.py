"""Request and resource ID helpers."""

from __future__ import annotations

import secrets


def request_id() -> str:
    """Return a short, URL-safe, unpredictable request identifier."""
    return secrets.token_urlsafe(12)


__all__ = ["request_id"]
