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


class ModelLicenseError(VisionServeXError):
    """Base for license-policy / commercial-safety gate errors."""


class ModelNotCommercialSafeError(ModelLicenseError):
    def __init__(self, model_id: str, status: str, hint: str = "") -> None:
        super().__init__(
            f"Model {model_id!r} is not commercial-safe (status={status!r}).",
            code="MODEL_NOT_COMMERCIAL_SAFE",
            hint=hint or f"visionservex models explain {model_id}",
            details={"model_id": model_id, "commercial_status": status},
            docs="docs/model_policy.md",
        )


class ModelLicenseRestrictedError(ModelLicenseError):
    def __init__(self, model_id: str, status: str, hint: str = "") -> None:
        super().__init__(
            f"Model {model_id!r} has a restricted license (status={status!r}) and is "
            "not enabled by default.",
            code="MODEL_LICENSE_RESTRICTED",
            hint=hint or f"visionservex models explain {model_id}",
            details={"model_id": model_id, "commercial_status": status},
            docs="docs/model_policy.md",
        )


class ModelAcknowledgementRequiredError(ModelLicenseError):
    def __init__(self, model_id: str, status: str, acknowledgement_text: str) -> None:
        super().__init__(
            f"Model {model_id!r} is restricted ({status}). You must explicitly pass "
            "use_mode and acknowledge_license_restrictions=True to use it.",
            code="MODEL_ACKNOWLEDGEMENT_REQUIRED",
            hint=(
                'VisionModel("'
                + model_id
                + '", use_mode="research", acknowledge_license_restrictions=True)'
            ),
            details={
                "model_id": model_id,
                "commercial_status": status,
                "acknowledgement_text": acknowledgement_text,
            },
            docs="docs/model_policy.md",
        )


class ModelRequiresBYOCheckpointError(ModelLicenseError):
    def __init__(self, model_id: str) -> None:
        super().__init__(
            f"Model {model_id!r} requires you to supply your own checkpoint (BYO).",
            code="MODEL_REQUIRES_BYO_CHECKPOINT",
            hint=f"Pass checkpoint=... (or --checkpoint) for {model_id!r}.",
            details={"model_id": model_id},
            docs="docs/byo_checkpoint.md",
        )


class ModelLicenseReviewRequiredError(ModelLicenseError):
    def __init__(self, model_id: str) -> None:
        super().__init__(
            f"Model {model_id!r} has an unclear/unverified license and is blocked by default.",
            code="MODEL_LICENSE_REVIEW_REQUIRED",
            hint=f"visionservex models explain {model_id}",
            details={"model_id": model_id},
            docs="docs/model_policy.md",
        )


class ModelAGPLRestrictedError(ModelLicenseError):
    def __init__(self, model_id: str) -> None:
        super().__init__(
            f"Model {model_id!r} is AGPL/copyleft or enterprise-license-restricted and is "
            "not in the commercial-safe set.",
            code="MODEL_AGPL_RESTRICTED",
            hint=f"visionservex models explain {model_id}",
            details={"model_id": model_id},
            docs="docs/model_policy.md",
        )


class ModelUseModeNotAllowedError(ModelLicenseError):
    def __init__(self, model_id: str, use_mode: str, allowed: tuple[str, ...]) -> None:
        super().__init__(
            f"use_mode={use_mode!r} is not allowed for {model_id!r}. Allowed: {list(allowed)}.",
            code="MODEL_USE_MODE_NOT_ALLOWED",
            hint=f"Use one of {list(allowed)} for {model_id!r}.",
            details={
                "model_id": model_id,
                "use_mode": use_mode,
                "allowed_use_modes": list(allowed),
            },
            docs="docs/model_policy.md",
        )


__all__ = [
    "DeviceUnavailableError",
    "EngineDependencyError",
    "ExternalModelError",
    "InputNotFoundError",
    "ManualModelError",
    "ModelAGPLRestrictedError",
    "ModelAcknowledgementRequiredError",
    "ModelLicenseError",
    "ModelLicenseRestrictedError",
    "ModelLicenseReviewRequiredError",
    "ModelMissingWeightsError",
    "ModelNotCommercialSafeError",
    "ModelNotFoundError",
    "ModelRequiresBYOCheckpointError",
    "ModelUseModeNotAllowedError",
    "SidecarNotRunningError",
    "TaskNotSupportedError",
    "VisionServeXError",
]
