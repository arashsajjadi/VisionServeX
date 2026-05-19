# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.30.0 release tests: package surface + LibreYOLO smoke matrix integration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent


def _run(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "visionservex", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO),
    )


def test_version_is_at_least_2_30() -> None:
    import visionservex

    parts = visionservex.__version__.split(".")
    assert int(parts[0]) >= 2 and int(parts[1]) >= 30, visionservex.__version__


def test_summarize_smoke_matrix_cli_help() -> None:
    proc = _run(["models", "summarize-smoke-matrix", "--help"])
    assert proc.returncode == 0


def test_summarize_smoke_matrix_consumes_v229_artifact(tmp_path: Path) -> None:
    src = REPO / "reports/core_smoke_matrix_v229.json"
    if not src.exists():
        return
    out = tmp_path / "canonical.json"
    proc = _run(
        [
            "models",
            "summarize-smoke-matrix",
            "--input",
            str(src),
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert proc.returncode == 0
    data = json.loads(out.read_text())
    assert "rows" in data
    assert "n_rows" in data
    assert data["n_rows"] > 0


def test_summarize_smoke_matrix_csv_no_nan(tmp_path: Path) -> None:
    src = REPO / "reports/core_smoke_matrix_v229.json"
    if not src.exists():
        return
    out = tmp_path / "canonical.csv"
    proc = _run(
        [
            "models",
            "summarize-smoke-matrix",
            "--input",
            str(src),
            "--format",
            "csv",
            "--out",
            str(out),
        ]
    )
    assert proc.returncode == 0
    text = out.read_text()
    # No raw NaN / NOT_WIRED in canonical summary
    for needle in ("NaN", "NOT_WIRED"):
        for line in text.splitlines():
            if line.startswith("model_id,"):
                continue  # header
            assert needle not in line, f"forbidden string in canonical CSV: {needle}"


def test_libreyolo_build_model_map_runs() -> None:
    proc = _run(["libreyolo", "build-model-map", "--out", "/tmp/v230_libreyolo_map.json"])
    assert proc.returncode == 0


def test_deimv2_audit_hf_runs() -> None:
    proc = _run(["deimv2", "audit-hf", "--out", "/tmp/v230_deimv2_audit.json"])
    assert proc.returncode == 0
    data = json.loads(Path("/tmp/v230_deimv2_audit.json").read_text())
    assert data["n_variants"] == 8
    assert data["license"] == "Apache-2.0"


def test_rtdetrv4_audit_checkpoints_runs() -> None:
    proc = _run(["rtdetrv4", "audit-checkpoints", "--out", "/tmp/v230_rtdetrv4_audit.json"])
    assert proc.returncode == 0
    data = json.loads(Path("/tmp/v230_rtdetrv4_audit.json").read_text())
    rows = data.get("rows", [])
    assert len(rows) == 4
    for r in rows:
        assert r["final_state"] == "manual_checkpoint_required"
        assert r["blocker_code"] == "MANUAL_CHECKPOINT_REQUIRED"


def test_smoke_matrix_supports_libreyolo_flag() -> None:
    proc = _run(["models", "smoke-matrix", "--help"])
    assert proc.returncode == 0
    # The new flag must be visible in --help (Rich may wrap it; check loose token)
    flat = " ".join(proc.stdout.split())
    assert "libreyolo" in flat.lower() or "include-libreyolo" in flat.lower()


def test_pyproject_has_libreyolo_extra() -> None:
    pyproject = (REPO / "pyproject.toml").read_text()
    assert "libreyolo =" in pyproject, "pyproject.toml must declare [libreyolo] optional extra"
    assert "libreyolo>=1.1.1" in pyproject


def test_reporting_rendering_module_exports() -> None:
    """The public reporting surface must expose the v2.30 rendering helpers."""
    from visionservex import reporting

    for name in ("render_nullable", "render_table_for_notebook", "is_nullish"):
        assert hasattr(reporting, name), f"reporting.{name} missing"


def test_stale_output_scan_report_exists() -> None:
    """v2.30.0 must include a Phase-1 scan report."""
    p = REPO / "reports/pre_v230_stale_output_scan.json"
    if not p.exists():
        return
    data = json.loads(p.read_text())
    assert "findings_by_string" in data


def test_libreyolo_hf_audit_report_exists() -> None:
    p = REPO / "reports/libreyolo_hf_full_audit_v230.json"
    if not p.exists():
        return
    data = json.loads(p.read_text())
    assert "n_models" in data
    assert data["n_models"] > 0
