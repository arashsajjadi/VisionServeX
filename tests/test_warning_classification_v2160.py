# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Phase 9 (v2.16.0): warning/stderr classification.

The v16 notebook treated every nonzero-returncode command as a failure and
every harmless HF ``chat_template`` warning as a problem. v2.16.0 ships a
canonical classifier so notebooks (and CI) can bucket each command into
``ok_clean``, ``ok_with_warning``, ``expected_blocker``, or one of the
``failed_*`` reasons.
"""

from __future__ import annotations

import json
from pathlib import Path

from visionservex.runtime.result_classifier import (
    STATUS_LEVELS,
    classify_command_result,
)


def test_returncode_zero_clean_stderr_is_ok_clean() -> None:
    r = classify_command_result(returncode=0, stdout='{"hello": 1}', stderr="")
    assert r.status == "ok_clean"
    assert r.warning_count == 0
    assert r.error_count == 0


def test_chat_template_warning_is_ok_with_warning() -> None:
    """HuggingFace `chat_template` warnings must not be classified as failures."""
    stderr = "Warning: No chat_template found for processor; using default."
    r = classify_command_result(returncode=0, stdout="{}", stderr=stderr)
    assert r.status == "ok_with_warning"
    assert r.warning_count == 1
    assert r.error_count == 0


def test_traceback_is_failed_runtime() -> None:
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "x.py", line 3, in <module>\n'
        '    raise RuntimeError("boom")\n'
        "RuntimeError: boom\n"
    )
    r = classify_command_result(returncode=1, stdout="", stderr=stderr)
    assert r.status == "failed_runtime"
    assert r.error_count == 1


def test_usage_error_is_failed_usage() -> None:
    stderr = "Usage: visionservex classify [OPTIONS]\nError: No such option: --bogus"
    r = classify_command_result(returncode=2, stdout="", stderr=stderr)
    assert r.status == "failed_usage"


def test_structured_expected_blocker_is_expected_blocker() -> None:
    payload = {"status": "expected_blocker", "code": "GATED_HF_AUTH_REQUIRED"}
    r = classify_command_result(returncode=0, stdout=json.dumps(payload), stderr="")
    assert r.status == "expected_blocker"
    assert r.structured_payload == payload


def test_strict_mode_promotes_expected_blocker_to_failed() -> None:
    payload = {"status": "expected_blocker", "code": "SIDECAR_REQUIRED"}
    r = classify_command_result(returncode=0, stdout=json.dumps(payload), stderr="", strict=True)
    assert r.status == "failed_runtime"


def test_anomalib_required_code_is_expected_blocker() -> None:
    payload = {"status": "expected_blocker", "code": "ANOMALIB_REQUIRED"}
    r = classify_command_result(returncode=0, stdout=json.dumps(payload), stderr="")
    assert r.status == "expected_blocker"


def test_missing_output_path_is_failed_output_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    r = classify_command_result(
        returncode=0,
        stdout="ok",
        stderr="",
        output_path=missing,
    )
    assert r.status == "failed_output_missing"
    assert r.artifact_checks["output_exists"] is False


def test_bad_json_artifact_is_failed_json_parse(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    r = classify_command_result(
        returncode=0,
        stdout="",
        stderr="",
        expect_json_at=bad,
    )
    assert r.status == "failed_json_parse"
    assert r.artifact_checks["json_parseable"] is False


def test_valid_json_artifact_is_ok_clean(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"x": 1}))
    r = classify_command_result(
        returncode=0,
        stdout="",
        stderr="",
        expect_json_at=good,
    )
    assert r.status == "ok_clean"
    assert r.artifact_checks["json_parseable"] is True


def test_unexplained_stderr_with_zero_returncode_is_warning() -> None:
    r = classify_command_result(returncode=0, stdout="", stderr="some weird notice")
    assert r.status == "ok_with_warning"


def test_nonzero_returncode_with_no_obvious_cause_is_failed_runtime() -> None:
    r = classify_command_result(returncode=137, stdout="", stderr="")
    assert r.status == "failed_runtime"


def test_all_statuses_are_in_STATUS_LEVELS() -> None:
    # Sanity: every status the function may produce is enumerated.
    assert set(STATUS_LEVELS) >= {
        "ok_clean",
        "ok_with_warning",
        "expected_blocker",
        "failed_usage",
        "failed_runtime",
        "failed_output_missing",
        "failed_json_parse",
    }
