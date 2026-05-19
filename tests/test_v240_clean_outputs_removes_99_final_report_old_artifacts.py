# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.40.0: ``visionservex notebook clean-outputs`` must delete old
``Final_Report_EXECUTED_v*.ipynb``, ``environment_v*.json``,
``coverage_cleanliness_v*.json``, and similar version-tagged artifacts
under ``notebook/99_final_report/``."""

from __future__ import annotations

import subprocess
import sys


def test_clean_outputs_removes_99_final_report_old_artifacts(tmp_path):
    fake = tmp_path / "notebook"
    final = fake / "99_final_report"
    final.mkdir(parents=True)
    (final / "reports").mkdir()
    (final / "Final_Report.ipynb").write_text("{}")
    # Old executed notebooks (must be deleted)
    for v in ("v234", "v235", "v236", "v237", "v238", "v2381"):
        (final / f"Final_Report_EXECUTED_{v}.ipynb").write_text("{}")
    # Old environment / cleanliness / consistency files (must be deleted)
    for name in (
        "environment_v235.json",
        "environment_v236.json",
        "coverage_cleanliness_v238.json",
        "v239_final_report_consistency.json",
        "v239_stale_final_table_audit.json",
        "quality_scan.json",
        "environment_report.json",
        "root_cleanliness_report.json",
    ):
        (final / "reports" / name).write_text("{}")
    # Preserved: source notebook + datasets dir
    (fake / "datasets").mkdir()
    (fake / "datasets" / "img.jpg").write_text("IMG")
    (fake / ".venv").mkdir()

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "visionservex.__main__",
            "notebook",
            "clean-outputs",
            "--root",
            str(fake),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    # Old artifacts must be gone
    for v in ("v234", "v235", "v236", "v237", "v238", "v2381"):
        assert not (final / f"Final_Report_EXECUTED_{v}.ipynb").exists(), (
            f"Final_Report_EXECUTED_{v}.ipynb should have been deleted"
        )
    for name in (
        "environment_v235.json",
        "environment_v236.json",
        "coverage_cleanliness_v238.json",
        "v239_final_report_consistency.json",
        "v239_stale_final_table_audit.json",
        "quality_scan.json",
    ):
        assert not (final / "reports" / name).exists(), f"{name} should have been deleted"

    # Preserved
    assert (final / "Final_Report.ipynb").exists()
    assert (fake / "datasets" / "img.jpg").exists()
    assert (fake / ".venv").exists()
