# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Model registry.

The registry is a typed, machine-readable directory of supported models. It
carries license, status, implementation maturity, hardware requirements, and
download metadata for every model.

Status taxonomy:
  stable        - integration tested in CI with MockEngine; engine fully wired.
  beta          - real backend implemented but still maturing.
  experimental  - integration partial; outputs may change.
  manual        - users must manually download/build before use.
  external      - upstream is API-gated or otherwise not self-hostable.
  stub          - registry entry only; no real backend wired in this build.

Implementation status (separate from project status):
  wired         - this build runs the real backend when dependencies are present.
  partial       - some code path is wired; expect rough edges.
  stub          - no real inference; calls fall back to MockEngine *only* when
                  the user opts in via allow_mock_fallback.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from importlib import resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

Task = Literal[
    "detect",
    "segment",
    "pose",
    "classify",
    "obb",
    "open_vocab_detect",
    "grounded_segment",
    "foundation_segment",
]

Status = Literal["stable", "beta", "experimental", "manual", "external", "stub", "optional"]
ImplementationStatus = Literal["wired", "partial", "stub"]
Difficulty = Literal["very_easy", "easy", "medium", "hard", "expert"]
DownloadType = Literal[
    "huggingface",
    "github_release",
    "direct_url",
    "manual",
    "external_api",
    "not_available",
    "synthetic",  # built-in mock — no download needed
    "package_managed",  # the engine package manages its own download (e.g. rfdetr)
]


class RegistryError(LookupError):
    """Raised when a registry lookup or registration fails."""


class ModelEntry(BaseModel):
    """Machine-readable metadata about a model.

    Fields are designed so a CLI or LLM agent can answer questions like
    "which models can run on my laptop?", "which models can I auto-download?",
    "what does this model require?" without inspecting code.
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    # Identity
    id: str
    display_name: str
    task: Task
    family: str
    backend: str = ""  # short label: pytorch / hf / onnx / mock / openmmlab / ...
    engine: str  # engines.registry key

    # Licensing
    license: str
    license_uncertain: bool = False
    license_notes: str | None = None
    commercial_use_notes: str | None = None
    upstream_url: str

    # Download
    download_type: DownloadType = "not_available"
    checkpoint_url: str | None = None
    hf_repo_id: str | None = None
    hf_revision: str | None = None
    checkpoint_filename: str | None = None
    checkpoint_sha256: str | None = None
    size_bytes: int | None = None
    requires_auth: bool = False
    requires_git_lfs: bool = False
    requires_custom_code: bool = False
    auto_download: bool = False

    # Software
    requires_optional_extra: bool = False
    install_extra: str | None = None
    requires_python: str = ">=3.10"
    auto_install_extra: bool = False  # always false; we never auto-pip-install
    implementation_notes: str | None = None
    implementation_status: ImplementationStatus = "stub"

    # Hardware
    supported_os: list[str] = Field(default_factory=lambda: ["linux", "darwin", "windows"])
    supported_devices: list[str] = Field(default_factory=lambda: ["cpu"])
    recommended_device: str = "cpu"
    fallback_device: str = "cpu"
    supported_precisions: list[str] = Field(default_factory=lambda: ["fp32"])
    minimum_vram_gb: float | None = None
    recommended_vram_gb: float | None = None
    minimum_ram_gb: float | None = None

    # Capabilities
    input_schema: dict = Field(default_factory=dict)
    output_schema: dict = Field(default_factory=dict)
    batch_support: bool = False
    streaming_support: bool = False
    export_support: list[str] = Field(default_factory=list)

    # UX hints
    difficulty: Difficulty = "medium"
    best_for: list[str] = Field(default_factory=list)
    not_good_for: list[str] = Field(default_factory=list)
    known_issues: list[str] = Field(default_factory=list)
    beginner_recommendation: bool = False

    # Project status
    status: Status = "stub"
    warnings: list[str] = Field(default_factory=list)
    notes: str | None = None

    # Back-compat alias
    @property
    def memory_hint_mb(self) -> int | None:
        """Approximate runtime memory hint in MiB (derived for legacy callers)."""
        if self.recommended_vram_gb is not None:
            return int(self.recommended_vram_gb * 1024)
        if self.minimum_vram_gb is not None:
            return int(self.minimum_vram_gb * 1024)
        return None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if not value or not all(ch.isalnum() or ch in "-_." for ch in value):
            raise ValueError(f"invalid model id: {value!r}")
        return value

    # ----- convenience -----

    def runs_on(self, device: str) -> bool:
        return device.lower() in {d.lower() for d in self.supported_devices}

    def is_downloadable(self) -> bool:
        return self.download_type in {
            "huggingface",
            "github_release",
            "direct_url",
            "synthetic",
            "package_managed",
        }

    def is_built_in(self) -> bool:
        return self.download_type == "synthetic"

    def needs_extra(self) -> bool:
        return self.requires_optional_extra and bool(self.install_extra)


class ModelRegistry:
    """Thread-safe in-memory registry of :class:`ModelEntry` records."""

    def __init__(self) -> None:
        self._entries: dict[str, ModelEntry] = {}
        self._lock = threading.RLock()

    # ----- loading -----

    def load_yaml(self, path: Path | str) -> None:
        with Path(path).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        self._load_data(data)

    def load_packaged(self) -> None:
        """Load the bundled ``models.yaml`` shipped inside the package."""
        ref = resources.files("visionservex.registry").joinpath("models.yaml")
        with resources.as_file(ref) as path:
            self.load_yaml(path)

    def _load_data(self, data: dict) -> None:
        items = data.get("models", [])
        if not isinstance(items, list):
            raise RegistryError("models.yaml: `models` must be a list")
        for raw in items:
            entry = ModelEntry.model_validate(raw)
            self.register(entry, replace=True)

    # ----- mutation -----

    def register(self, entry: ModelEntry, *, replace: bool = False) -> None:
        with self._lock:
            if not replace and entry.id in self._entries:
                raise RegistryError(f"model {entry.id!r} already registered")
            self._entries[entry.id] = entry

    def unregister(self, model_id: str) -> None:
        with self._lock:
            self._entries.pop(model_id, None)

    # ----- queries -----

    def get(self, model_id: str) -> ModelEntry:
        with self._lock:
            try:
                return self._entries[model_id]
            except KeyError as exc:
                raise RegistryError(
                    f"unknown model {model_id!r}. Run `visionservex list-models` to see options."
                ) from exc

    def has(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._entries

    def list(
        self,
        *,
        task: Task | None = None,
        status: Status | None = None,
        family: str | None = None,
        difficulty: Difficulty | None = None,
        auto_downloadable_only: bool = False,
        runs_on_device: str | None = None,
    ) -> list[ModelEntry]:
        with self._lock:
            entries = list(self._entries.values())
        if task is not None:
            entries = [e for e in entries if e.task == task]
        if status is not None:
            entries = [e for e in entries if e.status == status]
        if family is not None:
            entries = [e for e in entries if e.family == family]
        if difficulty is not None:
            entries = [e for e in entries if e.difficulty == difficulty]
        if auto_downloadable_only:
            entries = [e for e in entries if e.auto_download]
        if runs_on_device is not None:
            entries = [e for e in entries if e.runs_on(runs_on_device)]
        return sorted(entries, key=lambda e: (e.task, e.family, e.id))

    def __iter__(self) -> Iterable[ModelEntry]:
        return iter(self.list())

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


_default: ModelRegistry | None = None
_default_lock = threading.Lock()


def default_registry() -> ModelRegistry:
    """Return the process-wide registry, loading bundled models on first call."""
    global _default
    if _default is not None:
        return _default
    with _default_lock:
        if _default is None:
            reg = ModelRegistry()
            reg.load_packaged()
            _default = reg
    return _default


__all__ = [
    "Difficulty",
    "DownloadType",
    "ImplementationStatus",
    "ModelEntry",
    "ModelRegistry",
    "RegistryError",
    "Status",
    "Task",
    "default_registry",
]
