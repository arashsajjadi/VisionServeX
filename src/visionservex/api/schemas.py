# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Pydantic schemas for HTTP request and response bodies."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    public_mode: bool
    auth_enabled: bool


class VersionResponse(BaseModel):
    version: str
    python: str
    platform: str
    author: str = "Arash Sajjadi"


class ModelListItem(BaseModel):
    id: str
    display_name: str
    task: str
    family: str
    license: str
    status: str
    implementation_status: str
    engine: str
    backend: str
    difficulty: str
    auto_download: bool
    supported_devices: list[str]
    minimum_vram_gb: float | None = None
    recommended_vram_gb: float | None = None
    warnings: list[str] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    models: list[ModelListItem]


class PromptRequest(BaseModel):
    """JSON body for endpoints that accept prompts and a base64 image."""

    image_b64: str | None = Field(
        default=None,
        description="Base64-encoded image bytes. Mutually exclusive with `image_url`.",
    )
    image_url: str | None = Field(
        default=None,
        description="HTTPS URL of an image. Requires inputs.allow_url_inputs=true.",
    )
    prompts: list[str] = Field(default_factory=list)
    model_id: str
    options: dict[str, Any] = Field(default_factory=dict)


class UrlInput(BaseModel):
    image_url: str
    model_id: str
    options: dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    """Stable wire format for every prediction endpoint."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    request_id: str
    status: Literal["completed", "downloading", "queued", "failed"] = "completed"
    model_id: str
    task: str
    backend: str = ""
    device: str
    precision: str = "fp32"
    latency_ms: float
    model_loaded_from: str | None = None
    cache_path: str | None = None
    fallback_reason: str | None = None
    results: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobProgress(BaseModel):
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    percent: float = 0.0
    speed_bytes_per_sec: int = 0


class JobResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    request_id: str | None = None
    job_id: str
    kind: str
    model_id: str
    status: str
    message: str = ""
    progress: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    error: dict[str, Any] | None = None
    created_at: float
    updated_at: float


class DownloadingResponse(BaseModel):
    """Returned when an auto-pull starts and the client opted out of waiting."""

    model_config = ConfigDict(protected_namespaces=())
    request_id: str
    status: Literal["downloading"] = "downloading"
    model_id: str
    job_id: str
    message: str
    progress_url: str


__all__ = [
    "HealthResponse",
    "VersionResponse",
    "ModelListItem",
    "ModelListResponse",
    "PromptRequest",
    "UrlInput",
    "PredictionResponse",
    "JobProgress",
    "JobResponse",
    "DownloadingResponse",
]
