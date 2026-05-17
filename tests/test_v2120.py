# SPDX-License-Identifier: Apache-2.0
"""Tests for v2.12.0 audit infrastructure.

Covers the audit CLI (export-model-inventory, export-notebook-manifest, etc.),
license audit, model-zoo blockers --all, and docs/audit/ artifact consistency.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# audit export commands
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_audit_export_model_inventory_schema(tmp_path):
    from visionservex.cli.audit_commands import app

    out = tmp_path / "inventory.json"
    runner = CliRunner()
    result = runner.invoke(app, ["export-model-inventory", "--out", str(out)])
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert payload["n_models"] >= 100
    first = payload["models"][0]
    for key in (
        "model_id",
        "family",
        "task",
        "expected_load_mode",
        "smoke_command",
        "eligible_for_detection_ap",
        "eligible_for_ultralytics_comparison",
        "eligible_for_classification_benchmark",
        "eligible_for_segmentation_metric",
        "eligible_for_embedding_demo",
        "requires_sidecar",
        "recommended_colab_mode",
        "notebook_section",
    ):
        assert key in first, f"model row missing {key!r}"


@pytest.mark.fast
def test_audit_export_notebook_manifest_schema(tmp_path):
    from visionservex.cli.audit_commands import app

    out = tmp_path / "manifest.json"
    runner = CliRunner()
    result = runner.invoke(app, ["export-notebook-manifest", "--out", str(out)])
    assert result.exit_code == 0, result.output
    m = json.loads(out.read_text())
    for key in (
        "package",
        "families",
        "models",
        "commands",
        "benchmark_groups",
        "ultralytics_comparison",
        "expected_blockers",
        "sidecars",
        "optional_extras",
        "license_risks",
        "notebook_sections",
    ):
        assert key in m, f"manifest missing top-level key {key!r}"
    assert len(m["models"]) >= 100
    assert len(m["notebook_sections"]) >= 10
    assert m["package"]["version"]


@pytest.mark.fast
def test_audit_no_model_has_ultralytics_compare_and_non_detection():
    """Only closed-set detection models may be Ultralytics-comparable."""
    from visionservex.audit.builder import export_model_inventory

    inv = export_model_inventory()
    for m in inv["models"]:
        if m.get("eligible_for_ultralytics_comparison"):
            assert m.get("eligible_for_detection_ap"), (
                f"{m['model_id']} eligible for Ultralytics comparison but not detection AP"
            )
        # Embedding-only families must not be Ultralytics-comparable
        if m["family"] in ("dinov2", "clip", "siglip", "siglip2"):
            assert not m.get("eligible_for_ultralytics_comparison"), (
                f"embedding family {m['family']} must not be Ultralytics-comparable"
            )


@pytest.mark.fast
def test_audit_every_model_has_exactly_one_notebook_section():
    from visionservex.audit.builder import export_model_inventory

    inv = export_model_inventory()
    for m in inv["models"]:
        assert m.get("notebook_section"), f"{m['model_id']} missing notebook_section"
        assert isinstance(m["notebook_section"], str)


@pytest.mark.fast
def test_audit_bundle_creates_all_files(tmp_path):
    from visionservex.audit.builder import build_audit_bundle

    written = build_audit_bundle(str(tmp_path))
    expected = [
        "model_inventory_json",
        "feature_inventory_json",
        "command_inventory_json",
        "notebook_manifest_json",
        "benchmark_plan_md",
        "ultralytics_comparison_json",
        "expected_blockers_md",
        "model_test_matrix_csv",
        "model_inventory_md",
        "notebook_manifest_md",
    ]
    for key in expected:
        assert key in written, f"bundle missing artifact {key!r}"
        assert Path(written[key]).exists(), f"artifact file missing: {written[key]}"


@pytest.mark.fast
def test_model_test_matrix_csv_has_correct_columns(tmp_path):
    from visionservex.audit.builder import build_audit_bundle

    written = build_audit_bundle(str(tmp_path))
    csv_path = written["model_test_matrix_csv"]
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    assert "model_id" in fieldnames
    assert "eligible_for_ultralytics_comparison" in fieldnames
    assert "smoke_command" in fieldnames


# ---------------------------------------------------------------------------
# license commands
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_license_audit_returns_structured_json():
    from visionservex.cli.license_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["audit", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "n_entries" in payload
    assert "risk_summary" in payload
    assert "core_safe_verdict" in payload
    assert "PASS" in payload["core_safe_verdict"]


@pytest.mark.fast
def test_license_audit_gpl_not_in_core():
    from visionservex.cli.license_commands import _LICENSE_RISK_TABLE

    gpl = [e for e in _LICENSE_RISK_TABLE if "gpl" in e.get("risk", "")]
    for entry in gpl:
        assert entry.get("route") in ("do_not_add", "non_core_license_optional"), (
            f"{entry['model_or_lib']} is GPL but route={entry['route']!r}"
        )


# ---------------------------------------------------------------------------
# model-zoo blockers --all
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_model_zoo_blockers_all_flag():
    from visionservex.cli.model_zoo_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["blockers", "--all", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, list)
    assert len(payload) >= 5
    for entry in payload:
        assert "family" in entry
        assert "blocker_certainty" in entry
        assert entry["blocker_certainty"] >= 90


# ---------------------------------------------------------------------------
# docs/audit artifacts committed
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_docs_audit_directory_populated():
    audit_dir = ROOT / "docs" / "audit"
    assert audit_dir.exists()
    expected_files = [
        "visionservex_model_inventory.json",
        "visionservex_notebook_input_manifest.json",
        "visionservex_model_test_matrix.csv",
        "visionservex_benchmark_plan.md",
        "visionservex_ultralytics_comparison_plan.json",
        "visionservex_expected_blockers.md",
    ]
    for fname in expected_files:
        assert (audit_dir / fname).exists(), f"audit artifact missing: {fname}"


@pytest.mark.fast
def test_notebook_manifest_json_is_parseable():
    p = ROOT / "docs" / "audit" / "visionservex_notebook_input_manifest.json"
    if not p.exists():
        pytest.skip("Notebook manifest not yet generated in docs/audit/")
    m = json.loads(p.read_text())
    assert len(m.get("models", [])) >= 100
    assert len(m.get("notebook_sections", [])) >= 10


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_version_is_2120():
    import visionservex

    assert visionservex.__version__ == "2.12.0"
