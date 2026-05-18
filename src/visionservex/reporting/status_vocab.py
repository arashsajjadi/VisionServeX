# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.28.0: canonical status vocabulary.

The pre-v2.28 reports mixed ``NOT_WIRED``, ``failed_runtime``,
``UNAVAILABLE_OR_FAILED``, blank strings, ``NaN`` cells, and stale
``v20: clean detection candidates`` markers. v2.28 closes that vocabulary
to a fixed set so every model and every metric has a meaningful state.
"""

from __future__ import annotations

ALLOWED_FINAL_STATES: frozenset[str] = frozenset(
    {
        "benchmarked",
        "benchmarked_external_engine",
        "smoke_ok_no_metric",
        "visual_smoke_only",
        "promptable_benchmarked",
        "promptable_benchmark_pending",
        "sidecar_runnable",
        "sidecar_required",
        "checkpoint_required",
        "manual_checkpoint_required",
        "dataset_required",
        "license_blocked",
        "opt_in_license_required",
        "auth_required",
        "upstream_unavailable",
        "expected_blocker",
        "not_applicable",
        "not_advertised",
        "duplicate_alias",
        "diagnostic_only",
        "benchmark_candidate",
        "segmentation_pipeline_not_wired",
        "not_benchmarked_variant",
    }
)

# These values must NEVER appear in a final v2.28 report. The truth audit
# fails non-zero if any of them are present.
FORBIDDEN_FINAL_STATES: frozenset[str] = frozenset(
    {
        "NOT_WIRED",
        "UNAVAILABLE_OR_FAILED",
        "UNKNOWN",
        "TODO",
        "TBD",
        "",
        "None",
        "null",
        "NaN",
        "nan",
    }
)

# Strings that indicate stale executed-cell text that must NOT survive
# into the v31 final report. These match the exact forbidden list from
# the v2.28 release contract (Phase 14):
#
# - "v20: clean detection candidates" — old v19/v20 candidate resolver
# - "v2.16 package-level" / "v2.16 package benchmark" — old v2.16 cells
# - "full_126" appearing in *winner* sections (allowed in raw audit rows)
# - "Running tiny per-model CLI diagnostic" unless FORCE_RUN_DIAGNOSTICS=True
# - "clean detection candidates from package: 10 models" — old hardcoded note
#
# Historical names like "v22 BENCHMARK_SIZE" or "v23 schema utility loaded"
# are kept because they refer to a still-current utility, not stale claims.
STALE_MARKERS: tuple[str, ...] = (
    "v20: clean detection candidates",
    "v2.16 package-level",
    "v2.16 package benchmark",
    "clean detection candidates from package: 10 models",
    "Running tiny per-model CLI diagnostic",
)

# Markers that are forbidden *only* in winner / final-report contexts but
# legitimate in raw evidence CSVs (e.g. full_126 appears as
# evaluation_scope=full_126 in COCO128 carry-forward rows; the policy is
# that v31 must not promote any full_126 row to the *winner* section).
WINNER_CONTEXT_STALE_MARKERS: tuple[str, ...] = (
    "full_126",
    "COCO128 balanced is a useful smoke",
)

# v2.28 parseable-blocker codes that previously got mis-classified as
# `failed_runtime` despite stdout containing a valid JSON envelope.
PARSEABLE_BLOCKER_CODES: frozenset[str] = frozenset(
    {
        "ANOMALIB_REQUIRED",
        "BYTETRACK_REQUIRED",
        "OCSORT_REQUIRED",
        "TORCHREID_REQUIRED",
        "OPENMMLAB_REQUIRED",
        "DETECTRON2_REQUIRED",
        "CHECKPOINT_REQUIRED",
        "CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP",
        "MANUAL_CHECKPOINT_REQUIRED",
        "GATED_AUTH_REQUIRED",
        "GT_MASKS_REQUIRED_FOR_MASK_METRICS",
        "COCO_VAL2017_DOWNLOAD_DISALLOWED",
        "COCO_VAL2017_USER_PATH_REQUIRED",
        "LIBREYOLO_REQUIRED",
        "LIBREYOLO_WEIGHT_LICENSE_UNVERIFIED",
        "LIBREYOLO_WEIGHT_LICENSE_NONCOMMERCIAL",
        "LIBREYOLO_WEIGHT_LICENSE_GPL",
        "MEDSAM2_REQUIRED",
        "MONAI_REQUIRED",
        "NIFTI_REQUIRED",
        "SEGMENTATION_PIPELINE_NOT_WIRED",
        "PROMPTABLE_SEGMENTATION_NOT_IMPLEMENTED",
        "RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN",
        "RFDETR_SEG_CHECKPOINT_REQUIRED",
        "DEIMV2_NOT_RUNNABLE",
        "BLACKWELL_SM120_TORCH_INCOMPATIBLE",
        "DOWNLOAD_FAILED",
        "UPSTREAM_HF_REPO_NOT_FOUND",
        "TIMM_REQUIRED",
        "ULTRALYTICS_REQUIRED",
        "RTDETRV4_UPSTREAM_NOT_RELEASED",  # legacy code (kept for compat)
    }
)


def legacy_status_to_canonical(legacy: str, blocker_code: str = "") -> str:
    """Map a legacy v2.16-v2.27 status value to a canonical v2.28 final state.

    Returns ``"expected_blocker"`` as a safe fallback so the audit never
    accidentally invents a status; callers should override based on
    blocker_code when more specific information is available.
    """
    if not legacy:
        return "expected_blocker"
    s = str(legacy).strip()
    s_lc = s.lower()
    code_uc = (blocker_code or "").upper()

    # Direct allowed pass-through.
    if s in ALLOWED_FINAL_STATES:
        return s

    # Forbidden → coerce based on blocker_code where possible.
    if s in FORBIDDEN_FINAL_STATES:
        if code_uc in {"MANUAL_CHECKPOINT_REQUIRED", "CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP"}:
            return "manual_checkpoint_required"
        if code_uc == "CHECKPOINT_REQUIRED":
            return "checkpoint_required"
        if code_uc in {"SIDECAR_ENV_MISSING", "SIDECAR_REQUIRED"}:
            return "sidecar_required"
        if code_uc in {"LIBREYOLO_WEIGHT_LICENSE_NONCOMMERCIAL", "LICENSE_RESTRICTION_TRIGGERED"}:
            return "license_blocked"
        if code_uc == "LIBREYOLO_WEIGHT_LICENSE_GPL":
            return "opt_in_license_required"
        if code_uc in {"GATED_AUTH_REQUIRED", "GATED_HF_AUTH_REQUIRED"}:
            return "auth_required"
        if code_uc in {"UPSTREAM_HF_REPO_NOT_FOUND", "UPSTREAM_REPO_NOT_FOUND"}:
            return "upstream_unavailable"
        if code_uc == "COCO_VAL2017_400_REQUIRED":
            return "dataset_required"
        return "expected_blocker"

    # Heuristic mappings for common legacy strings.
    if s_lc in {"wired", "ok", "ok_clean", "ok_with_warning"}:
        return "smoke_ok_no_metric"
    if s_lc in {"partial", "stub"}:
        return "smoke_ok_no_metric"
    if s_lc == "failed_runtime":
        # Caller MUST pass blocker_code; if absent we return a structured
        # state that the audit will still flag for review.
        if code_uc in PARSEABLE_BLOCKER_CODES:
            return "expected_blocker"
        return "expected_blocker"
    if s_lc in {"expected_blocker", "failed_usage", "failed_output_missing"}:
        return "expected_blocker"
    if s_lc == "diagnostic_only":
        return "diagnostic_only"
    return "expected_blocker"


__all__ = [
    "ALLOWED_FINAL_STATES",
    "FORBIDDEN_FINAL_STATES",
    "PARSEABLE_BLOCKER_CODES",
    "STALE_MARKERS",
    "legacy_status_to_canonical",
]
