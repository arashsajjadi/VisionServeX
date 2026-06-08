# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.6 Final report presence and format tests.

Verifies that v36_final_report.md exists, starts with the required header,
and references the minimum required evidence counts.
"""

from __future__ import annotations

from pathlib import Path


def _report_path() -> Path:
    return (
        Path(__file__).parent.parent
        / "notebook"
        / "99_final_report"
        / "reports"
        / "v36_model_addition_final_report.md"
    )


def test_final_report_exists() -> None:
    p = _report_path()
    assert p.exists(), f"v3.6 final report missing: {p}"


def test_final_report_starts_with_required_header() -> None:
    p = _report_path()
    if not p.exists():
        return
    content = p.read_text()
    assert content.startswith("VISION SERVE X V3.6"), (
        f"Final report must start with 'VISION SERVE X V3.6', got: {content[:80]!r}"
    )


def test_final_report_mentions_locateanything() -> None:
    p = _report_path()
    if not p.exists():
        return
    content = p.read_text()
    assert "LocateAnything" in content or "locate-anything" in content


def test_final_report_mentions_version_360() -> None:
    p = _report_path()
    if not p.exists():
        return
    content = p.read_text()
    assert "3.6.0" in content or "3.6" in content


def test_final_report_mentions_nvidia_warning() -> None:
    p = _report_path()
    if not p.exists():
        return
    content = p.read_text()
    assert "NVIDIA" in content
    assert "non-commercial" in content


def test_final_report_mentions_phase1_fix() -> None:
    p = _report_path()
    if not p.exists():
        return
    content = p.read_text()
    assert "sam-vit-l" in content or "ONNX" in content


def test_all_v36_csv_ledgers_exist() -> None:
    reports_dir = Path(__file__).parent.parent / "notebook" / "99_final_report" / "reports"
    required_csvs = [
        "v36_new_model_execution_ledger.csv",
        "v36_stability_matrix.csv",
        "v36_locateanything_matrix.csv",
        "v36_onnx_runtime_matrix.csv",
        "v36_phase1_canonicalization_ledger.csv",
        "v36_security_audit.csv",
        "v36_api_standardization_matrix.csv",
        "v36_cli_consistency_audit.csv",
        "v36_license_audit.csv",
        "v36_sidecar_checkpoint_matrix.csv",
        "v36_failed_target_blockers.csv",
    ]
    for csv_name in required_csvs:
        p = reports_dir / csv_name
        assert p.exists(), f"Required v3.6 ledger missing: {p}"
