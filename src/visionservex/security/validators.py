"""Input validators."""

from __future__ import annotations

from pathlib import Path

from visionservex.config import InputsConfig, LimitsConfig
from visionservex.utils.images import ImageValidationError, open_safe
from visionservex.utils.paths import PathTraversalError, safe_join


class InputValidationError(ValueError):
    pass


def validate_mime_type(content_type: str | None, inputs: InputsConfig) -> str:
    if not content_type:
        raise InputValidationError("missing Content-Type")
    primary = content_type.split(";")[0].strip().lower()
    if primary not in {m.lower() for m in inputs.allowed_mime_types}:
        raise InputValidationError(f"unsupported content type: {primary!r}")
    return primary


def validate_image_bytes(data: bytes, limits: LimitsConfig) -> None:
    """Verify the bytes decode to an image within configured limits."""
    if len(data) == 0:
        raise InputValidationError("empty image upload")
    if len(data) > limits.max_upload_bytes:
        raise InputValidationError(
            f"upload {len(data)} bytes exceeds max_upload_bytes {limits.max_upload_bytes}"
        )
    try:
        open_safe(data, max_pixels=limits.max_image_pixels, max_dim=limits.max_image_dim)
    except ImageValidationError as exc:
        raise InputValidationError(str(exc)) from exc


def validate_path_input(path: str, *, root: Path, inputs: InputsConfig) -> Path:
    """Resolve a user-supplied local path inside ``root`` if local paths are allowed."""
    if not inputs.allow_local_paths:
        raise InputValidationError("local file path input is disabled by configuration")
    try:
        return safe_join(root, path)
    except PathTraversalError as exc:
        raise InputValidationError(str(exc)) from exc


__all__ = [
    "InputValidationError",
    "validate_image_bytes",
    "validate_mime_type",
    "validate_path_input",
]
