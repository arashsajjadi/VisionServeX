"""Stable API error envelope."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status
from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    hint: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class ApiError(HTTPException):
    """HTTPException subclass that produces a stable error envelope.

    Endpoints raise this; the server's exception handler renders it as
    ``{"request_id": ..., "error": {...}}``.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        hint: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.error_code = code
        self.error_message = message
        self.error_hint = hint
        self.error_details = details or {}


# Common helpers
def not_found(code: str, message: str, *, hint: str = "") -> ApiError:
    return ApiError(status.HTTP_404_NOT_FOUND, code, message, hint=hint)


def bad_request(code: str, message: str, *, hint: str = "") -> ApiError:
    return ApiError(status.HTTP_400_BAD_REQUEST, code, message, hint=hint)


def unauthorized(message: str = "authentication required", *, hint: str = "") -> ApiError:
    return ApiError(
        status.HTTP_401_UNAUTHORIZED,
        "UNAUTHENTICATED",
        message,
        hint=hint or "send Authorization: Bearer <api-key>",
    )


def forbidden(message: str, *, hint: str = "") -> ApiError:
    return ApiError(status.HTTP_403_FORBIDDEN, "FORBIDDEN", message, hint=hint)


def too_large(message: str, *, hint: str = "") -> ApiError:
    return ApiError(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "REQUEST_TOO_LARGE", message, hint=hint)


def unprocessable(code: str, message: str, *, hint: str = "") -> ApiError:
    # 422; avoid touching the legacy ``HTTP_422_UNPROCESSABLE_ENTITY`` name
    # unconditionally to suppress a Starlette deprecation warning.
    return ApiError(422, code, message, hint=hint)


def busy(message: str, *, hint: str = "") -> ApiError:
    return ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "BUSY", message, hint=hint)


__all__ = [
    "ErrorBody",
    "ApiError",
    "not_found",
    "bad_request",
    "unauthorized",
    "forbidden",
    "too_large",
    "unprocessable",
    "busy",
]
