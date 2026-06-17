# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Typed exceptions for VisionServeX.

Every user-facing error carries a machine-readable ``code``, a human-readable
``message``, an actionable ``hint``, and optional ``details``.
"""

from __future__ import annotations

from typing import Any


class VisionServeXError(RuntimeError):
    """Base typed exception for VisionServeX.

    Attributes:
        code:    Machine-readable error code (e.g. ``"MODEL_NOT_FOUND"``).
        message: Human-readable description.
        hint:    Exact command or action to fix the problem.
        details: Optional structured metadata.
        docs:    Optional docs URL.
        alternative: Optional alternative model id to suggest.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "VISIONSERVEX_ERROR",
        hint: str = "",
        details: dict[str, Any] | None = None,
        docs: str | None = None,
        alternative: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.details = details or {}
        self.docs = docs
        self.alternative = alternative

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "details": self.details,
            "docs": self.docs,
            "alternative": self.alternative,
        }

    def __str__(self) -> str:
        parts = [f"[{self.code}] {self.message}"]
        if self.hint:
            parts.append(f"Fix: {self.hint}")
        if self.alternative:
            parts.append(f"Try: {self.alternative}")
        return "\n".join(parts)


class ModelNotFoundError(VisionServeXError):
    def __init__(self, model_id: str) -> None:
        super().__init__(
            f"Model {model_id!r} is not in the registry.",
            code="MODEL_NOT_FOUND",
            hint="Run: visionservex list-models  OR  visionservex recommend --task detect --simple",
            docs="docs/model_zoo.md",
        )


class InputNotFoundError(VisionServeXError):
    def __init__(self, path: str) -> None:
        super().__init__(
            f"Input file not found: {path!r}",
            code="INPUT_NOT_FOUND",
            hint="Check the file path and try again.",
        )


class DeviceUnavailableError(VisionServeXError):
    def __init__(self, device: str, reason: str = "") -> None:
        super().__init__(
            f"Device {device!r} is not available.{' ' + reason if reason else ''}",
            code="DEVICE_UNAVAILABLE",
            hint=(
                "Run `visionservex gpu doctor` for diagnostics, or use `--device cpu`."
                if "cuda" in device.lower()
                else "Run `visionservex devices` to see available devices."
            ),
            docs="docs/device_selection.md",
        )


class ModelMissingWeightsError(VisionServeXError):
    def __init__(self, model_id: str) -> None:
        super().__init__(
            f"Model weights for {model_id!r} are not cached locally.",
            code="MODEL_MISSING",
            hint=f"visionservex pull {model_id}",
            docs="docs/model_downloads.md",
        )


class SidecarNotRunningError(VisionServeXError):
    def __init__(self, model_id: str) -> None:
        super().__init__(
            f"The OpenMMLab sidecar is required for {model_id!r} but is not running.",
            code="SIDECAR_NOT_RUNNING",
            hint=("visionservex openmmlab docker-build && visionservex openmmlab docker-run"),
            docs="docs/openmmlab_expert_models.md",
        )


class ExternalModelError(VisionServeXError):
    def __init__(self, model_id: str, alternative: str | None = None) -> None:
        super().__init__(
            f"Model {model_id!r} is external/API-gated and cannot be self-hosted.",
            code="EXTERNAL_MODEL",
            hint=f"visionservex info {model_id}  — see upstream for access",
            alternative=alternative,
        )


class ManualModelError(VisionServeXError):
    def __init__(
        self, model_id: str, instructions: str = "", alternative: str | None = None
    ) -> None:
        super().__init__(
            f"Model {model_id!r} requires manual installation.",
            code="MANUAL_MODEL",
            hint=instructions or f"visionservex info {model_id}",
            docs="docs/model_downloads.md",
            alternative=alternative,
        )


class EngineDependencyError(VisionServeXError):
    def __init__(self, model_id: str, extra: str) -> None:
        super().__init__(
            f"Required dependency is not installed for {model_id!r}.",
            code="ENGINE_UNAVAILABLE",
            hint=f"pip install 'visionservex[{extra}]'",
            docs="docs/installation.md",
        )


class TaskNotSupportedError(VisionServeXError):
    """Raised when a task-specific method (classify/embed/segment/correspond) is
    called on a model whose registered task does not support it (v3.17.0)."""

    def __init__(self, model_id: str, method: str, task: str, hint: str = "") -> None:
        super().__init__(
            f"{method}() is not supported for model {model_id!r} (task={task!r}).",
            code="TASK_NOT_SUPPORTED",
            hint=hint or f"visionservex info {model_id}",
            docs="docs/model_syntax_matrix.md",
        )


__all__ = [
    "DeviceUnavailableError",
    "EngineDependencyError",
    "ExternalModelError",
    "InputNotFoundError",
    "ManualModelError",
    "ModelMissingWeightsError",
    "ModelNotFoundError",
    "SidecarNotRunningError",
    "TaskNotSupportedError",
    "VisionServeXError",
]
