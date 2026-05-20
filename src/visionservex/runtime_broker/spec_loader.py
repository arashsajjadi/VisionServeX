# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Loader and validator for ``runtime_specs.yaml``.

The runtime broker depends entirely on the spec file. If the file is missing,
malformed, or violates the expected schema, the broker must refuse to operate
with a structured :class:`SpecLoadError` so callers can show a precise message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "RuntimeSpec",
    "SpecLoadError",
    "load_specs",
]


_REQUIRED_RUNTIME_IDS: frozenset[str] = frozenset(
    {
        "core_py311",
        "rtdetrv4_py311_torch",
        "codetr_openmmlab_py310",
        "internimage_dcnv3_py310",
        "obb_rtmdetr2_py310",
        "obb_mmrotate_legacy_py39",
        "pose_mmpose_py310",
        "oneformer_natten_py310",
        "maskdino_detectron2_py310",
        "tracking_bytetrack_py310",
        "promptable_edgesam_py310",
        "medical_medsam2_py310",
        "seem_xdecoder_container",
        "license_gate_runtime",
        "auth_gate_runtime",
        "registry_audit_runtime",
    }
)


class SpecLoadError(Exception):
    """Raised when ``runtime_specs.yaml`` cannot be loaded or fails validation."""


@dataclass
class RuntimeSpec:
    """In-memory view of a runtime entry."""

    id: str
    description: str
    env_type: str
    python_version: str | None
    torch_version: str | None
    cuda_version: str | None
    pip_packages: list[str]
    conda_packages: list[str]
    git_repos: list[dict[str, Any]]
    custom_ops: list[str]
    checkpoint_sources: list[dict[str, Any]]
    license_gate: str | None
    auth_gate: str | None
    output_adapter: str | None
    smoke_command: list[str]
    contract_command: list[str]
    benchmark_command: list[str]
    known_failure_modes: list[dict[str, Any]] = field(default_factory=list)
    fallback_runtime: str | None = None
    supported_models: list[str] = field(default_factory=list)

    @property
    def needs_sidecar_env(self) -> bool:
        return self.env_type in {"conda", "venv", "docker"}

    @property
    def is_external(self) -> bool:
        return self.env_type == "http"


def _default_spec_path() -> Path:
    """Return the path to the packaged ``runtime_specs.yaml``."""

    return Path(__file__).parent / "runtime_specs.yaml"


def load_specs(path: Path | str | None = None) -> dict[str, RuntimeSpec]:
    """Load and validate the runtime specs.

    Returns a mapping ``runtime_id -> RuntimeSpec``. Raises :class:`SpecLoadError`
    on any validation failure. Validation checks:

    * file parses as YAML;
    * top-level ``runtimes`` is a list;
    * every required runtime id from :data:`_REQUIRED_RUNTIME_IDS` is present;
    * every entry has the mandatory keys;
    * env_type is one of {conda, venv, docker, subprocess, http};
    * fallback_runtime, if set, references a known runtime id;
    * supported_models lists are unique within the file.
    """

    spec_path = Path(path) if path is not None else _default_spec_path()
    if not spec_path.exists():
        raise SpecLoadError(f"runtime_specs.yaml not found at {spec_path}")

    try:
        with spec_path.open() as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise SpecLoadError(f"runtime_specs.yaml parse error: {exc}") from exc

    if not isinstance(data, dict) or "runtimes" not in data:
        raise SpecLoadError("runtime_specs.yaml missing top-level 'runtimes' key")

    runtimes_raw = data.get("runtimes")
    if not isinstance(runtimes_raw, list):
        raise SpecLoadError("'runtimes' must be a list")

    specs: dict[str, RuntimeSpec] = {}
    seen_models: dict[str, str] = {}

    for idx, entry in enumerate(runtimes_raw):
        if not isinstance(entry, dict):
            raise SpecLoadError(f"runtime entry at index {idx} is not a mapping")

        try:
            spec = RuntimeSpec(
                id=entry["id"],
                description=entry["description"],
                env_type=entry["env_type"],
                python_version=entry.get("python_version"),
                torch_version=entry.get("torch_version"),
                cuda_version=entry.get("cuda_version"),
                pip_packages=list(entry.get("pip_packages") or []),
                conda_packages=list(entry.get("conda_packages") or []),
                git_repos=list(entry.get("git_repos") or []),
                custom_ops=list(entry.get("custom_ops") or []),
                checkpoint_sources=list(entry.get("checkpoint_sources") or []),
                license_gate=entry.get("license_gate"),
                auth_gate=entry.get("auth_gate"),
                output_adapter=entry.get("output_adapter"),
                smoke_command=list(entry.get("smoke_command") or []),
                contract_command=list(entry.get("contract_command") or []),
                benchmark_command=list(entry.get("benchmark_command") or []),
                known_failure_modes=list(entry.get("known_failure_modes") or []),
                fallback_runtime=entry.get("fallback_runtime"),
                supported_models=list(entry.get("supported_models") or []),
            )
        except KeyError as exc:
            raise SpecLoadError(
                f"runtime entry at index {idx} missing required key: {exc!s}"
            ) from exc

        if spec.env_type not in {"conda", "venv", "docker", "subprocess", "http"}:
            raise SpecLoadError(f"runtime '{spec.id}' has invalid env_type: {spec.env_type!r}")

        if spec.id in specs:
            raise SpecLoadError(f"duplicate runtime id: {spec.id}")

        for model_id in spec.supported_models:
            if model_id in seen_models:
                raise SpecLoadError(
                    f"model '{model_id}' is claimed by both '{seen_models[model_id]}' "
                    f"and '{spec.id}'"
                )
            seen_models[model_id] = spec.id

        specs[spec.id] = spec

    missing = _REQUIRED_RUNTIME_IDS - specs.keys()
    if missing:
        raise SpecLoadError(f"runtime_specs.yaml missing required runtimes: {sorted(missing)}")

    for spec in specs.values():
        if spec.fallback_runtime and spec.fallback_runtime not in specs:
            raise SpecLoadError(
                f"runtime '{spec.id}' fallback '{spec.fallback_runtime}' is undefined"
            )

    return specs
