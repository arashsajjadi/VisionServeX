# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""The generic VisionServeX sidecar request/response protocol.

A sidecar request normalises a model call; a sidecar response normalises into the
same public schema VisionServeX uses in-process, so downstream code cannot tell a
sidecar-served model from a native one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from visionservex.sidecars.errors import SidecarRequestInvalid, SidecarResponseInvalid

_METHODS = frozenset({"predict", "classify", "segment", "pose", "obb", "vlm", "embed", "detect"})


@dataclass(kw_only=True)
class SidecarRequest:
    """A normalised request forwarded to a sidecar over HTTP (multipart)."""

    model_id: str
    task: str
    method: str
    image: bytes | str | None = None  # raw bytes or a path the caller resolves
    text: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""

    def validate(self) -> SidecarRequest:
        if not self.model_id:
            raise SidecarRequestInvalid("model_id is required")
        if self.method not in _METHODS:
            raise SidecarRequestInvalid(
                f"unknown method {self.method!r}; one of {sorted(_METHODS)}"
            )
        if not self.task:
            raise SidecarRequestInvalid("task is required")
        return self

    def form_fields(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "task": self.task,
            "method": self.method,
            "text": self.text,
            "request_id": self.request_id,
        }


@dataclass(kw_only=True)
class SidecarResponse:
    """A normalised sidecar response. ``payload`` is a public-schema dict."""

    model_id: str
    task: str
    payload: dict[str, Any]

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SidecarResponse:
        if not isinstance(data, dict):
            raise SidecarResponseInvalid(f"expected a JSON object, got {type(data).__name__}")
        mid = data.get("model_id")
        if not mid:
            raise SidecarResponseInvalid("response missing model_id")
        return cls(model_id=mid, task=data.get("task", ""), payload=data)


__all__ = ["SidecarRequest", "SidecarResponse"]
