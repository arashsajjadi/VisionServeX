# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Classify a CLI command result into a single canonical status.

The v16 notebook ran every CLI command, then tried to bucket each into pass /
warning / failure / expected-blocker. It made many wrong calls because:

- ``returncode == 0`` was treated as pass even when stderr printed a
  HuggingFace ``chat_template`` warning.
- A 404 for an optional ``preprocessor_config.json`` was treated as a failure
  even though the model still ran.
- Real tracebacks were grouped with ordinary stderr noise.

This module turns those signals into a single ``status`` field shared by
every CLI: ``ok_clean``, ``ok_with_warning``, ``expected_blocker``,
``failed_usage``, ``failed_runtime``, ``failed_output_missing``,
``failed_json_parse``, ``failed_artifact_invalid``.

The classifier is conservative: it never returns ``ok_clean`` if there is
any unrecognised stderr content, but it also never returns ``failed`` for
known harmless HuggingFace/Transformers noise.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "STATUS_LEVELS",
    "ClassifiedResult",
    "classify_command_result",
    "classify_stderr_lines",
]

STATUS_LEVELS = (
    "ok_clean",
    "ok_with_warning",
    "expected_blocker",
    "failed_usage",
    "failed_runtime",
    "failed_output_missing",
    "failed_json_parse",
    "failed_artifact_invalid",
)

# Patterns that are harmless and should not count as failures.
# Each pattern is a regex matched against a single stderr line.
_HARMLESS_STDERR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"chat_template", re.IGNORECASE),
    re.compile(r"no chat template", re.IGNORECASE),
    re.compile(r"You are using the default legacy behaviour", re.IGNORECASE),
    re.compile(r"FutureWarning", re.IGNORECASE),
    re.compile(r"DeprecationWarning", re.IGNORECASE),
    re.compile(r"UserWarning", re.IGNORECASE),
    re.compile(r"`processor_config.json` not found", re.IGNORECASE),
    re.compile(r"preprocessor_config\.json", re.IGNORECASE),
    re.compile(r"HTTPError 404 .* preprocessor", re.IGNORECASE),
    re.compile(r"Some weights of .* were not initialized", re.IGNORECASE),
    re.compile(r"You should probably TRAIN this model", re.IGNORECASE),
    re.compile(r"INFO:", re.IGNORECASE),
    re.compile(r"DEBUG:", re.IGNORECASE),
    re.compile(r"^\s*$"),
)

# Patterns that indicate a real failure.
_FATAL_STDERR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^Traceback", re.MULTILINE),
    re.compile(r"^\s*File \".*\", line \d+", re.MULTILINE),
    re.compile(r"^[A-Z][A-Za-z]*Error:", re.MULTILINE),
    re.compile(r"Segmentation fault", re.IGNORECASE),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
)

# Patterns that indicate a usage / argument-parsing failure.
_USAGE_STDERR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^Usage:", re.MULTILINE),
    re.compile(r"No such option", re.IGNORECASE),
    re.compile(r"Got unexpected extra argument", re.IGNORECASE),
    re.compile(r"Missing argument", re.IGNORECASE),
    re.compile(r"Invalid value for", re.IGNORECASE),
)

# Codes that the package already emits inside a structured payload to mark
# "this command did not run but it was expected — install something".
EXPECTED_BLOCKER_CODES = frozenset(
    [
        "DEPENDENCY_REQUIRED",
        "CHECKPOINT_REQUIRED",
        "GATED_AUTH_REQUIRED",
        "GATED_HF_AUTH_REQUIRED",
        "SIDECAR_REQUIRED",
        "EXTERNAL_API_REQUIRED",
        "ANOMALIB_REQUIRED",
        "MONAI_REQUIRED",  # v2.19.0
        "NIFTI_REQUIRED",  # v2.19.0
        # v2.21.0: structured blockers surfaced by the v23 notebook run that
        # were previously mis-bucketed as failed_runtime.
        "BYTETRACK_REQUIRED",
        "TORCHREID_REQUIRED",
        "OCSORT_REQUIRED",
        "TOTAL_SEGMENTATOR_REQUIRED",
        "TOTALSEGMENTATOR_REQUIRED",
        "NNUNET_REQUIRED",
        "OPENMMLAB_REQUIRED",
        "DETECTRON2_REQUIRED",
        "MMDET_REQUIRED",
        "MMROTATE_REQUIRED",
        "MMSEGMENTATION_REQUIRED",
        "MEDSAM2_REQUIRED",
        "DEIM_REQUIRED",
        "DEIMV2_REQUIRED",
        "RTDETRV4_REQUIRED",
        "RFDETR_PLUS_LICENSE_BLOCKED",
        "NON_CORE_LICENSE_OPT_IN_REQUIRED",
        # generic source-audit blockers
        "MODEL_SOURCE_NOT_AVAILABLE",
        "MODEL_NOT_RUNNABLE_IN_THIS_BUILD",
        "BOX_PROMPTS_REQUIRED",  # v2.19.0
        "LABELS_REQUIRED_FOR_METRICS",  # v2.19.0
        "DOTA_OR_OBB_LABELS_REQUIRED",  # v2.19.0
        "AERIAL_LABELS_REQUIRED",  # v2.19.0
        "GT_TRACKS_OR_QUERY_LABELS_REQUIRED",  # v2.19.0
        "GT_MASKS_REQUIRED",  # v2.19.0
        "BENCHMARK_NOT_IMPLEMENTED",  # v2.19.0
        "TASK_NOT_SUPPORTED",  # v2.19.0
        "SHARED_MODEL_CONCURRENCY_NOT_SUPPORTED",  # v2.19.0
        "CONCURRENCY_RESOURCE_BLOCKED",  # v2.19.0
        "MODEL_NOT_RUNNABLE",
        "UNAVAILABLE_WITH_REASON",
        "OPENCV_REQUIRED",
        "TRACKER_UNAVAILABLE",
        "REID_UNAVAILABLE",
        "MODEL_LOAD_FAILED",
    ]
)


@dataclass
class ClassifiedResult:
    """Outcome of :func:`classify_command_result`."""

    status: str
    returncode: int
    warning_count: int = 0
    warnings: list[str] = field(default_factory=list)
    error_count: int = 0
    errors: list[str] = field(default_factory=list)
    stderr_summary: str = ""
    stdout_summary: str = ""
    artifact_checks: dict[str, Any] = field(default_factory=dict)
    structured_payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "returncode": self.returncode,
            "warning_count": self.warning_count,
            "warnings": list(self.warnings),
            "error_count": self.error_count,
            "errors": list(self.errors),
            "stderr_summary": self.stderr_summary,
            "stdout_summary": self.stdout_summary,
            "artifact_checks": dict(self.artifact_checks),
            "structured_payload": self.structured_payload,
        }


def classify_stderr_lines(stderr: str) -> tuple[list[str], list[str]]:
    """Split stderr into (warnings, errors). Each element is a single line."""
    if not stderr:
        return [], []
    warnings: list[str] = []
    errors: list[str] = []

    # Fatal regexes look at the *whole* stderr (Traceback spans lines).
    text = stderr
    is_fatal = any(p.search(text) for p in _FATAL_STDERR_PATTERNS)
    if is_fatal:
        errors.append(stderr[:1000])
        return warnings, errors

    for line in stderr.splitlines():
        if not line.strip():
            continue
        if any(p.search(line) for p in _HARMLESS_STDERR_PATTERNS):
            warnings.append(line.strip())
            continue
        # Anything else is an unexplained warning.
        warnings.append(line.strip())
    return warnings, errors


def _parse_structured_stdout(stdout: str) -> dict[str, Any] | None:
    """If stdout is a single JSON object, return it; else None."""
    s = stdout.strip()
    if not s:
        return None
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def classify_command_result(
    *,
    returncode: int,
    stdout: str,
    stderr: str,
    output_path: Path | str | None = None,
    expect_json_at: Path | str | None = None,
    strict: bool = False,
) -> ClassifiedResult:
    """Classify one CLI invocation into a single canonical status.

    Args:
        returncode: ``subprocess.CompletedProcess.returncode``.
        stdout, stderr: captured streams.
        output_path: optional artifact (image/video/JSON) that must exist.
        expect_json_at: if set, the file must parse as JSON.
        strict: if True, expected_blocker is treated as a failure
            (used by CI gates that want to fail on every blocker).

    Returns:
        :class:`ClassifiedResult`.
    """
    warnings_, errors_ = classify_stderr_lines(stderr)
    structured = _parse_structured_stdout(stdout)
    structured_code = (structured or {}).get("code")
    structured_status = (structured or {}).get("status")

    artifact_checks: dict[str, Any] = {}

    # Output artifact existence
    if output_path is not None:
        p = Path(output_path)
        artifact_checks["output_path"] = str(p)
        artifact_checks["output_exists"] = p.exists()
        artifact_checks["output_size_bytes"] = p.stat().st_size if p.exists() else 0

    json_parse_failed = False
    if expect_json_at is not None:
        p = Path(expect_json_at)
        artifact_checks["json_path"] = str(p)
        artifact_checks["json_exists"] = p.exists()
        if p.exists():
            try:
                json.loads(p.read_text())
                artifact_checks["json_parseable"] = True
            except json.JSONDecodeError as exc:
                artifact_checks["json_parseable"] = False
                artifact_checks["json_error"] = str(exc)[:200]
                json_parse_failed = True

    stderr_summary = stderr[:200] if stderr else ""
    stdout_summary = stdout[:200] if stdout else ""

    def _bag(status: str) -> ClassifiedResult:
        return ClassifiedResult(
            status=status,
            returncode=returncode,
            warning_count=len(warnings_),
            warnings=warnings_,
            error_count=len(errors_),
            errors=errors_,
            stderr_summary=stderr_summary,
            stdout_summary=stdout_summary,
            artifact_checks=artifact_checks,
            structured_payload=structured,
        )

    # 1. Hard runtime failures (traceback / segfault).
    if errors_:
        return _bag("failed_runtime")

    # 2. Usage errors.
    if any(p.search(stderr) for p in _USAGE_STDERR_PATTERNS):
        return _bag("failed_usage")

    # 3. Structured payload says expected_blocker → expected_blocker (unless strict).
    if structured_status == "expected_blocker" or (
        structured_code and structured_code in EXPECTED_BLOCKER_CODES
    ):
        if strict:
            return _bag("failed_runtime")
        return _bag("expected_blocker")

    # 4. JSON parse failure on a path we expected to be valid JSON.
    if json_parse_failed:
        return _bag("failed_json_parse")

    # 5. Missing artifact (only check if caller asked for one).
    if output_path is not None and not artifact_checks.get("output_exists"):
        return _bag("failed_output_missing")

    # 6. Non-zero return without an obvious cause.
    if returncode != 0:
        return _bag("failed_runtime")

    # 7. Returncode 0 but stderr had unexplained warnings.
    if warnings_:
        return _bag("ok_with_warning")

    return _bag("ok_clean")
