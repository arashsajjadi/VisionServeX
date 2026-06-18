# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Typed errors for the generic VisionServeX sidecar layer.

All sidecar failures map onto these so callers get an exact, redacted reason —
never a raw stack trace and never a leaked token.
"""

from __future__ import annotations

import re

# Redact anything that looks like an HF token in error text before it propagates.
_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{6,}")


def redact(text: str | None) -> str:
    if not text:
        return ""
    return _TOKEN_RE.sub("hf_***REDACTED", str(text))


class SidecarError(Exception):
    """Base class for sidecar failures. ``code`` is a stable machine string."""

    code = "SIDECAR_ERROR"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(redact(message))
        if code:
            self.code = code
        self.message = redact(message)


class SidecarNotConfigured(SidecarError):
    """The sidecar is disabled (no URL / extra / env var set)."""

    code = "SIDECAR_NOT_CONFIGURED"


class SidecarUnavailable(SidecarError):
    """The sidecar URL is set but the health check failed (not running)."""

    code = "SIDECAR_UNAVAILABLE"


class SidecarTimeout(SidecarError):
    """The sidecar did not respond within the deadline."""

    code = "SIDECAR_TIMEOUT"


class SidecarRequestInvalid(SidecarError):
    """The request did not satisfy the sidecar protocol schema."""

    code = "SIDECAR_REQUEST_INVALID"


class SidecarResponseInvalid(SidecarError):
    """The sidecar returned a payload that did not match the expected schema."""

    code = "SIDECAR_RESPONSE_INVALID"


__all__ = [
    "SidecarError",
    "SidecarNotConfigured",
    "SidecarRequestInvalid",
    "SidecarResponseInvalid",
    "SidecarTimeout",
    "SidecarUnavailable",
    "redact",
]
