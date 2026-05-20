# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Canonical result schema and structured blocker codes for the runtime broker.

Every adapter under ``visionservex.runtime_broker.adapters.*`` maps a sidecar's
native output to :class:`CanonicalResult`. Every failure carries a code from
:data:`BROKER_BLOCKER_CODES` so the result-classifier can match on a fixed
vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Broker-level codes plus every code declared by an individual runtime spec.
# The spec_loader merges runtime-specific codes into this set at load time
# (the static list below covers the broker's own error surfaces).
BROKER_BLOCKER_CODES: set[str] = {
    # broker control flow
    "UNKNOWN_MODEL_ID",
    "RUNTIME_NOT_FOUND_FOR_MODEL",
    "RUNTIME_SPEC_INVALID",
    "RUNTIME_NOT_PREPARED",
    "RUNTIME_PREPARE_TIMEOUT",
    "RUNTIME_PREPARE_FAILED",
    "OUTPUT_ADAPTER_NOT_FOUND",
    "OUTPUT_ADAPTER_RAISED",
    "BROKER_DRY_RUN_NO_EXECUTE",
    "BROKER_RESOURCE_GUARD_BLOCKED",
    # license/auth
    "LICENSE_OPT_IN_NOT_PROVIDED",
    "LICENSE_UNCONFIGURED_FOR_MODEL",
    "AUTH_TOKEN_NOT_PROVIDED",
    "AUTH_TOS_NOT_ACCEPTED",
    "API_KEY_NOT_PROVIDED",
    "API_ENDPOINT_UNREACHABLE",
    "API_QUOTA_EXCEEDED",
    # checkpoints / repos
    "CHECKPOINT_PULL_FAILED",
    "CHECKPOINT_PULL_REQUIRES_MANUAL_STEP",
    "REPO_CLONE_FAILED",
    "REPO_REF_NOT_FOUND",
    # core runtime
    "CORE_CHECKPOINT_NOT_FOUND",
    # codetr
    "CODETR_REPO_CLONE_FAILED",
    "CODETR_CONFIG_NOT_FOUND",
    "CODETR_CHECKPOINT_NOT_FOUND",
    "CODETR_HF_PULL_FAILED",
    "CODETR_GDRIVE_PULL_FAILED",
    "CODETR_STATE_DICT_MISMATCH",
    "CODETR_OUTPUT_ADAPTER_MISSING",
    "CODETR_CUDA_OOM",
    # internimage
    "DCNV3_BUILD_FAILED",
    "DCNV3_CUDA_VERSION_UNSUPPORTED",
    "INTERNIMAGE_HF_MODEL_NOT_FOUND",
    "INTERNIMAGE_REMOTE_CODE_FAILED",
    "INTERNIMAGE_CHECKPOINT_NOT_FOUND",
    "INTERNIMAGE_OUTPUT_ADAPTER_MISSING",
    # obb / rtmdetr2
    "RTMDETR2_REPO_CLONE_FAILED",
    "RTMDETR2_CONFIG_NOT_FOUND",
    "RTMDETR2_CHECKPOINT_NOT_FOUND",
    "RTMDETR2_IMPORT_FAILED",
    "MMYOLO_ROTATED_CONFIG_NOT_FOUND",
    "MMROTATE_1X_ENV_FAILED",
    "OBB_OUTPUT_ADAPTER_MISSING",
    "OBB_PROXY_NOT_REAL_MODEL",
    # oneformer / natten
    "NATTEN_API_MISMATCH_UNPATCHABLE",
    "TRANSFORMERS_DINAT_VERSION_INCOMPATIBLE",
    "ONEFORMER_CHECKPOINT_NOT_FOUND",
    "ONEFORMER_OUTPUT_ADAPTER_MISSING",
    # bytetrack
    "BYTETRACK_REPO_CLONE_FAILED",
    "LAP_PREINSTALL_FAILED",
    "BYTETRACK_IMPORT_FAILED",
    "TRACKING_SAMPLE_VIDEO_MISSING",
    "TRACKING_OUTPUT_ADAPTER_MISSING",
    # edgesam
    "EDGESAM_REPO_CLONE_FAILED",
    "EDGESAM_CHECKPOINT_NOT_FOUND",
    "EDGESAM_IMPORT_FAILED",
    "EDGESAM_OUTPUT_ADAPTER_MISSING",
    # medsam2
    "MEDSAM2_REPO_CLONE_FAILED",
    "MEDSAM2_CHECKPOINT_NOT_FOUND",
    "MEDSAM2_DATA_FORMAT_REQUIRED",
    "MEDSAM2_OUTPUT_ADAPTER_MISSING",
    # seem
    "SEEM_CONTAINER_BUILD_FAILED",
    "SEEM_REPO_CLONE_FAILED",
    "XDECODER_REPO_CLONE_FAILED",
    "OPENMPI_MISSING",
    "SEEM_CHECKPOINT_NOT_FOUND",
    "SEEM_OUTPUT_ADAPTER_MISSING",
    # maskdino
    "DETECTRON2_BUILD_FAILED",
    "MASKDINO_REPO_CLONE_FAILED",
    "MASKDINO_CHECKPOINT_NOT_FOUND",
    "MASKDINO_CONFIG_NOT_FOUND",
    "MASKDINO_CPU_CONTRACT_FAILED",
    "MASKDINO_OUTPUT_ADAPTER_MISSING",
    # registry audit
    "REGISTRY_AUDIT_DEPRECATED",
    "REGISTRY_AUDIT_WRONG_ENTRY",
    # mmpose (defensive)
    "MMPOSE_IMPORT_FAILED",
    # rtdetrv4
    "RTDETRV4_CHECKPOINT_DOWNLOAD_FAILED",
}


@dataclass(frozen=True)
class BrokerBlocker:
    """Structured failure descriptor returned by the broker."""

    code: str
    message: str
    runtime_id: str | None = None
    model_id: str | None = None
    next_action: str | None = None
    exception_type: str | None = None
    exception_tail: str | None = None


@dataclass
class CanonicalDetection:
    """Canonical bounding-box detection row."""

    xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str | None = None


@dataclass
class CanonicalOBB:
    """Canonical oriented bounding box (5-parameter form: cx, cy, w, h, angle_rad)."""

    cxcywha: tuple[float, float, float, float, float]
    score: float
    class_id: int
    class_name: str | None = None


@dataclass
class CanonicalMask:
    """Canonical instance/semantic mask reference.

    The mask itself is referenced by path so adapters do not need to round-trip
    multi-MB arrays through the broker.
    """

    mask_path: str
    class_id: int
    score: float | None = None
    class_name: str | None = None


@dataclass
class CanonicalResult:
    """Canonical result envelope returned by every adapter."""

    model_id: str
    runtime_id: str
    task: str
    detections: list[CanonicalDetection] = field(default_factory=list)
    obb: list[CanonicalOBB] = field(default_factory=list)
    masks: list[CanonicalMask] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    blocker: BrokerBlocker | None = None

    @property
    def ok(self) -> bool:
        return self.blocker is None
