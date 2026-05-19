# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.39.0: expanded blocker taxonomy with detailed per-model diagnostics.

This module extends :mod:`visionservex.reporting.status_vocab` with the
fine-grained blocker codes required by the v2.39 plan. Each unresolved
model row in the coverage ledger must carry one of these codes (or a
benchmark/demo/smoke/contract passed state) — never a vague
``expected_blocker``.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------- Dependency / install ----------
DEPENDENCY_CODES: frozenset[str] = frozenset(
    {
        "PYTHON_VERSION_UNSUPPORTED",
        "TORCH_VERSION_UNSUPPORTED",
        "CUDA_VERSION_UNSUPPORTED",
        "PACKAGE_MISSING",
        "PACKAGE_VERSION_CONFLICT",
        "CUSTOM_OP_BUILD_FAILED",
        "NATTEN_REQUIRED",
        "NATTEN_BUILD_FAILED",
        "DETECTRON2_BUILD_FAILED",
        "MMCV_BUILD_FAILED",
        "OPENMPI_REQUIRED",
        "FLORENCE2_TRANSFORMERS_VERSION_REQUIRED",
        "DEPENDENCY_REQUIRED",
    }
)

# ---------- Checkpoint / config ----------
CHECKPOINT_CODES: frozenset[str] = frozenset(
    {
        "CHECKPOINT_REQUIRED",
        "MANUAL_CHECKPOINT_REQUIRED",
        "CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP",
        "CHECKPOINT_DOWNLOAD_FAILED",
        "CHECKPOINT_INVALID",
        "CHECKPOINT_STATE_DICT_MISMATCH",
        "CHECKPOINT_DOWNLOADED",
        "CONFIG_REQUIRED",
        "CONFIG_NOT_FOUND",
        "CONFIG_MODEL_MISMATCH",
        "GOOGLE_DRIVE_AUTH_OR_QUOTA",
        "GDOWN_FAILED",
    }
)

# ---------- Loader / output ----------
LOADER_CODES: frozenset[str] = frozenset(
    {
        "LOADER_MISSING",
        "IMPORT_FAILED",
        "MODEL_INIT_FAILED",
        "INFERENCE_FAILED",
        "OUTPUT_SCHEMA_UNKNOWN",
        "OUTPUT_ADAPTER_MISSING",
        "CATEGORY_MAPPING_MISSING",
        "MASK_SCHEMA_MISMATCH",
        "BOX_SCHEMA_MISMATCH",
    }
)

# ---------- External ----------
EXTERNAL_CODES: frozenset[str] = frozenset(
    {
        "AUTH_REQUIRED",
        "HF_TOKEN_REQUIRED",
        "GATED_AUTH_REQUIRED",
        "GATED_HF_AUTH_REQUIRED",
        "API_KEY_REQUIRED",
        "LICENSE_ACCEPTANCE_REQUIRED",
        "OPT_IN_LICENSE_REQUIRED",
        "LICENSE_RESTRICTION_TRIGGERED",
        "UPSTREAM_DEPRECATED",
        "WRONG_REGISTRY_ENTRY",
        "UPSTREAM_404",
        "UPSTREAM_HF_REPO_NOT_FOUND",
        "DOWNLOAD_FAILED_RETRYABLE",
        "DOWNLOAD_FAILED",
        "DATASET_REQUIRED",
        "BENCHMARK_GT_REQUIRED",
    }
)

# ---------- Hardware ----------
HARDWARE_CODES: frozenset[str] = frozenset(
    {
        "CUDA_OOM",
        "CUDA_KERNEL_LAUNCH_FAILED",
        "BLACKWELL_UNSUPPORTED",
        "BLACKWELL_SM120_TORCH_INCOMPATIBLE",
        "LEGACY_CUDA_UNSUPPORTED",
        "MASKDINO_LEGACY_CUDA_BLACKWELL_UNSUPPORTED",
        "TENSORRT_UNSAFE_ON_5080",
    }
)

# ---------- RT-DETRv4 specific ----------
RTDETRV4_CODES: frozenset[str] = frozenset(
    {
        "RTDETRV4_ENV_CREATE_FAILED",
        "RTDETRV4_IMPORT_FAILED",
        "RTDETRV4_CONFIG_MISSING",
        "RTDETRV4_CHECKPOINT_INVALID",
        "RTDETRV4_STATE_DICT_MISMATCH",
        "RTDETRV4_OUTPUT_ADAPTER_MISSING",
        "RTDETRV4_CUDA_OOM",
    }
)

ALL_V239_BLOCKER_CODES: frozenset[str] = (
    DEPENDENCY_CODES
    | CHECKPOINT_CODES
    | LOADER_CODES
    | EXTERNAL_CODES
    | HARDWARE_CODES
    | RTDETRV4_CODES
)


@dataclass(frozen=True)
class BlockerDiagnostic:
    """A precise diagnostic for a single blocked model."""

    model_id: str
    blocker_code: str
    blocker_category: str  # dependency / checkpoint / loader / external / hardware
    blocker_subcategory: str = ""
    exact_exception_type: str = ""
    exact_error_message_tail: str = ""
    attempted_command: str = ""
    attempted_loader: str = ""
    attempted_backend: str = ""
    attempted_checkpoint_path: str = ""
    attempted_config_path: str = ""
    dependency_missing: str = ""
    required_package: str = ""
    required_version: str = ""
    sidecar_name: str = ""
    sidecar_python_version: str = ""
    sidecar_torch_version: str = ""
    cuda_required: str = ""
    cuda_observed: str = ""
    gpu_observed: str = ""
    auth_env_var: str = ""
    license_source: str = ""
    manual_fix_command: str = ""
    deep_research_needed: bool = False
    deep_research_prompt_path: str = ""
    evidence_artifact: str = ""


def categorize_blocker(blocker_code: str) -> str:
    """Return one of dependency / checkpoint / loader / external / hardware / rtdetrv4."""
    code = (blocker_code or "").strip().upper()
    if code in DEPENDENCY_CODES:
        return "dependency"
    if code in CHECKPOINT_CODES:
        return "checkpoint"
    if code in LOADER_CODES:
        return "loader"
    if code in EXTERNAL_CODES:
        return "external"
    if code in HARDWARE_CODES:
        return "hardware"
    if code in RTDETRV4_CODES:
        return "rtdetrv4"
    return "unclassified"


def is_v239_blocker_code(code: str) -> bool:
    """Return True if ``code`` is in the v2.39 expanded taxonomy."""
    return (code or "").strip().upper() in ALL_V239_BLOCKER_CODES


# Generic placeholder codes that the v2.39 ledger must not accept as final.
GENERIC_BLOCKER_CODES: frozenset[str] = frozenset(
    {
        "",
        "EXPECTED_BLOCKER",
        "STUB",
        "BLOCKED",
        "UNKNOWN",
        "TBD",
        "TODO",
        "UNAVAILABLE_OR_FAILED",
        "NOT_WIRED",
        "NaN",
        "None",
    }
)


def is_generic_blocker(code: str) -> bool:
    """Return True if ``code`` is a vague placeholder that v2.39 rejects."""
    return (code or "").strip().upper() in {c.upper() for c in GENERIC_BLOCKER_CODES}


__all__ = [
    "ALL_V239_BLOCKER_CODES",
    "CHECKPOINT_CODES",
    "DEPENDENCY_CODES",
    "EXTERNAL_CODES",
    "GENERIC_BLOCKER_CODES",
    "HARDWARE_CODES",
    "LOADER_CODES",
    "RTDETRV4_CODES",
    "BlockerDiagnostic",
    "categorize_blocker",
    "is_generic_blocker",
    "is_v239_blocker_code",
]
