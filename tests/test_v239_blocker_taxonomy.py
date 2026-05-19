# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.39.0: expanded blocker taxonomy."""

from __future__ import annotations

from visionservex.reporting.v239_blockers import (
    ALL_V239_BLOCKER_CODES,
    GENERIC_BLOCKER_CODES,
    BlockerDiagnostic,
    categorize_blocker,
    is_generic_blocker,
    is_v239_blocker_code,
)


def test_required_v239_codes_present() -> None:
    required = {
        # dependency
        "NATTEN_REQUIRED",
        "DETECTRON2_BUILD_FAILED",
        "MMCV_BUILD_FAILED",
        "OPENMPI_REQUIRED",
        "FLORENCE2_TRANSFORMERS_VERSION_REQUIRED",
        # checkpoint
        "CHECKPOINT_STATE_DICT_MISMATCH",
        "CHECKPOINT_DOWNLOADED",
        "GOOGLE_DRIVE_AUTH_OR_QUOTA",
        # loader
        "OUTPUT_ADAPTER_MISSING",
        "MASK_SCHEMA_MISMATCH",
        "BOX_SCHEMA_MISMATCH",
        # external
        "OPT_IN_LICENSE_REQUIRED",
        "WRONG_REGISTRY_ENTRY",
        "UPSTREAM_DEPRECATED",
        # hardware
        "BLACKWELL_UNSUPPORTED",
        "MASKDINO_LEGACY_CUDA_BLACKWELL_UNSUPPORTED",
        "TENSORRT_UNSAFE_ON_5080",
        # RT-DETRv4 family
        "RTDETRV4_ENV_CREATE_FAILED",
        "RTDETRV4_CHECKPOINT_INVALID",
        "RTDETRV4_OUTPUT_ADAPTER_MISSING",
    }
    for code in required:
        assert code in ALL_V239_BLOCKER_CODES, f"required v2.39 code {code!r} missing"


def test_categorize_blocker():
    assert categorize_blocker("NATTEN_REQUIRED") == "dependency"
    assert categorize_blocker("CHECKPOINT_DOWNLOADED") == "checkpoint"
    assert categorize_blocker("OUTPUT_ADAPTER_MISSING") == "loader"
    assert categorize_blocker("OPT_IN_LICENSE_REQUIRED") == "external"
    assert categorize_blocker("BLACKWELL_UNSUPPORTED") == "hardware"
    assert categorize_blocker("RTDETRV4_ENV_CREATE_FAILED") == "rtdetrv4"
    assert categorize_blocker("totally_made_up") == "unclassified"


def test_is_v239_blocker_code():
    assert is_v239_blocker_code("NATTEN_REQUIRED")
    assert not is_v239_blocker_code("WAT_IS_THIS")


def test_is_generic_blocker_rejects_vague_codes():
    for vague in GENERIC_BLOCKER_CODES:
        assert is_generic_blocker(vague), f"{vague!r} should be flagged as generic"
    assert not is_generic_blocker("CHECKPOINT_DOWNLOADED")
    assert not is_generic_blocker("OPT_IN_LICENSE_REQUIRED")


def test_blocker_diagnostic_dataclass():
    diag = BlockerDiagnostic(
        model_id="maskdino-r50-coco",
        blocker_code="MASKDINO_LEGACY_CUDA_BLACKWELL_UNSUPPORTED",
        blocker_category="hardware",
        sidecar_name="detectron2_detseg_py38",
        sidecar_python_version="3.8",
        sidecar_torch_version="1.9.0",
        cuda_required="cu111",
        cuda_observed="sm_120",
        gpu_observed="RTX 5080",
        manual_fix_command="see reports/v239_maskdino_doctor.json",
    )
    assert diag.blocker_category == "hardware"
    assert diag.sidecar_torch_version == "1.9.0"
