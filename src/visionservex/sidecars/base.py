# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Base helper for sidecar-backed engines.

A :class:`SidecarBackend` bundles a :class:`SidecarClient` with the env var that
enables it. An engine is *available* only when the sidecar is both configured and
its health check passes — otherwise the model stays hidden/blocked. This is the
single chokepoint that keeps sidecar models out of the default UI unless a healthy
sidecar is actually running.
"""

from __future__ import annotations

from typing import Any

from visionservex.sidecars.client import SidecarClient
from visionservex.sidecars.protocol import SidecarRequest


class SidecarBackend:
    """Bind a named sidecar to an enabling env var. Disabled unless that var is set."""

    def __init__(self, *, name: str, url_env: str, timeout: float = 60.0) -> None:
        self.name = name
        self.url_env = url_env
        self.timeout = timeout
        self._client: SidecarClient | None = None

    @property
    def client(self) -> SidecarClient:
        if self._client is None:
            self._client = SidecarClient.from_env(
                self.url_env, name=self.name, timeout=self.timeout
            )
        return self._client

    def configured(self) -> bool:
        return self.client.configured

    def available(self) -> bool:
        """Configured AND healthy — the only state in which a sidecar model is usable."""
        return self.client.configured and self.client.health()

    def forward(
        self,
        *,
        model_id: str,
        task: str,
        method: str,
        image_bytes: bytes,
        text: str = "",
        params: dict[str, Any] | None = None,
        request_id: str = "",
    ) -> dict[str, Any]:
        """Send a normalised request; return the normalised public-schema payload."""
        req = SidecarRequest(
            model_id=model_id,
            task=task,
            method=method,
            text=text,
            params=params or {},
            request_id=request_id,
        )
        return self.client.predict(req, image_bytes=image_bytes).payload


__all__ = ["SidecarBackend"]
