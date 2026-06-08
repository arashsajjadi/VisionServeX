# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7: release-readiness gate — version, report header, ledgers, guards."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent
R = ROOT / "notebook" / "99_final_report" / "reports"


def test_version_is_370():
    import visionservex

    # forward-compatible: must not regress below the v3.7 baseline
    assert tuple(int(x) for x in visionservex.__version__.split(".")[:2]) >= (3, 7)


def test_pyproject_version_370():
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert tuple(int(x) for x in data["project"]["version"].split(".")[:2]) >= (3, 7)


def test_required_ledgers_exist():
    for f in [
        "v37_post_v259_inventory.csv",
        "v37_sam_variant_matrix.csv",
        "v37_dino_variant_matrix.csv",
        "v37_new_model_execution_ledger.csv",
        "v37_license_decisions.csv",
        "v37_execution_summary.json",
    ]:
        assert (R / f).exists(), f"missing ledger {f}"


def test_final_report_header():
    rep = R / "v37_table_completion_final_report.md"
    assert rep.exists(), f"missing {rep}"
    first = rep.read_text().splitlines()[0]
    assert first.startswith("VISION SERVE X V3.7 TABLE-COMPLETION PRODUCTIZATION FINAL STATUS"), (
        f"bad header: {first!r}"
    )


def test_cli_entry_point_registered():
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["project"]["scripts"]["visionservex"] == "visionservex.cli.main:app"


def test_new_cli_modules_importable():
    import visionservex.cli.interactive_commands
    import visionservex.cli.segment_instances_commands
    import visionservex.interactive_runtime
    import visionservex.rfdetr_seg_runtime  # noqa: F401


def test_interactive_extra_in_pyproject():
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert "interactive-seg" in extras or "rfdetr" in extras
