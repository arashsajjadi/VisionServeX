# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Generic HTTP client for a VisionServeX model sidecar.

Local-only by default, strict timeout, health/version probes, token redaction in
every error path, and no persistence. A sidecar is *disabled* unless its URL env
var is explicitly set — the package never reaches out on its own.
"""

from __future__ import annotations

import os
from typing import Any

from visionservex.sidecars.errors import (
    SidecarNotConfigured,
    SidecarResponseInvalid,
    SidecarTimeout,
    SidecarUnavailable,
    redact,
)
from visionservex.sidecars.protocol import SidecarRequest, SidecarResponse


class SidecarClient:
    """Talks to one sidecar over HTTP. Construct from an env var via :meth:`from_env`."""

    def __init__(self, base_url: str | None, *, name: str, timeout: float = 30.0) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.name = name
        self.timeout = timeout

    @classmethod
    def from_env(cls, env_var: str, *, name: str, timeout: float = 30.0) -> SidecarClient:
        return cls(os.environ.get(env_var), name=name, timeout=timeout)

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def _require_configured(self) -> None:
        if not self.configured:
            raise SidecarNotConfigured(
                f"{self.name} sidecar is not configured (set its URL env var); model stays hidden"
            )

    def _httpx(self):
        try:
            import httpx  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SidecarUnavailable("httpx not installed for sidecar HTTP") from exc
        return httpx

    def health(self) -> bool:
        """True iff the sidecar is configured AND its /health returns 200."""
        if not self.configured:
            return False
        httpx = self._httpx()
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=min(5.0, self.timeout))
            return r.status_code == 200
        except Exception:
            return False

    def version(self) -> dict[str, Any]:
        self._require_configured()
        httpx = self._httpx()
        try:
            r = httpx.get(f"{self.base_url}/version", timeout=min(5.0, self.timeout))
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            raise SidecarUnavailable(f"{self.name} /version failed: {exc}") from None

    def predict(self, request: SidecarRequest, *, image_bytes: bytes) -> SidecarResponse:
        """POST a normalised request; return a normalised response. Never raises raw."""
        self._require_configured()
        request.validate()
        httpx = self._httpx()
        try:
            r = httpx.post(
                f"{self.base_url}/predict",
                data=request.form_fields(),
                files={"image": ("image.png", image_bytes, "application/octet-stream")},
                timeout=self.timeout,
            )
        except httpx.TimeoutException:
            raise SidecarTimeout(f"{self.name} predict timed out after {self.timeout}s") from None
        except Exception as exc:
            raise SidecarUnavailable(f"{self.name} predict failed: {exc}") from None
        if r.status_code != 200:
            raise SidecarResponseInvalid(
                f"{self.name} returned HTTP {r.status_code}: {redact(r.text)[:200]}"
            )
        return SidecarResponse.from_json(r.json())


__all__ = ["SidecarClient"]
