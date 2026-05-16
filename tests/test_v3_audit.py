# SPDX-License-Identifier: Apache-2.0
"""v3 release audit: README hygiene, model load matrix, CLI help sweep.

These tests enforce the v3 release rule that the public README must not
display internal readiness percentages, that every registry model
appears exactly once in the load matrix, and that every CLI subapp
loads its `--help` without crashing.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# README hygiene
# ---------------------------------------------------------------------------


_README_FORBIDDEN = (
    "functional_readiness",
    "operational_readiness",
    "blocker_certainty",
    "Readiness factor table",
    "remaining gap",
    # The header text from the v2.9 readiness table — moved to docs/.
    "Overall production readiness",
)


_README_REQUIRED_SECTIONS = (
    "## Quickstart",
    "## License",
)


@pytest.mark.fast
def test_readme_has_no_internal_readiness_percentages():
    body = (ROOT / "README.md").read_text()
    found = [token for token in _README_FORBIDDEN if token in body]
    assert not found, (
        f"README must not display internal readiness telemetry; found: {found}. "
        "Move that content to docs/release_readiness/."
    )


@pytest.mark.fast
def test_readme_contains_required_public_sections():
    body = (ROOT / "README.md").read_text()
    missing = [s for s in _README_REQUIRED_SECTIONS if s not in body]
    assert not missing, f"README missing required sections: {missing}"


@pytest.mark.fast
def test_readiness_docs_moved_to_release_readiness_dir():
    p = ROOT / "docs" / "release_readiness" / "v2.9.0.md"
    assert p.exists(), "Release readiness telemetry must live under docs/release_readiness/."
    body = p.read_text()
    # The percentage table must be in the relocated doc.
    assert "Overall production readiness" in body
    assert "Functional" in body
    assert "Operational" in body


# ---------------------------------------------------------------------------
# Model load matrix
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_load_matrix_lists_every_registry_model_exactly_once():
    from visionservex.cli.model_health_commands import _load_matrix_rows
    from visionservex.registry import default_registry

    rows = _load_matrix_rows()
    ids = [r["model_id"] for r in rows]
    reg_ids = [e.id for e in default_registry().list()]
    assert sorted(ids) == sorted(set(ids)), "load matrix contains duplicate model ids"
    assert sorted(ids) == sorted(reg_ids), (
        f"load matrix model count differs from registry: "
        f"{len(ids)} matrix vs {len(reg_ids)} registry"
    )


@pytest.mark.fast
def test_load_matrix_assigns_a_known_mode_to_every_row():
    from visionservex.cli.model_health_commands import _load_matrix_rows

    valid_modes = {
        "core_load",
        "optional_extra_load",
        "sidecar_validate",
        "gated_auth_validate",
        "non_core_license_validate",
        "external_api_validate",
        "unavailable_blocker_validate",
        "do_not_add_validate",
    }
    rows = _load_matrix_rows()
    bad = [
        (r["model_id"], r["expected_load_mode"])
        for r in rows
        if r["expected_load_mode"] not in valid_modes
    ]
    assert not bad, f"unknown expected_load_mode values: {bad}"


@pytest.mark.fast
def test_load_matrix_every_row_has_smoke_command():
    from visionservex.cli.model_health_commands import _load_matrix_rows

    rows = _load_matrix_rows()
    empty = [r["model_id"] for r in rows if not r["smoke_command"]]
    assert not empty, f"models missing smoke_command: {empty}"


@pytest.mark.fast
def test_load_matrix_cli_writes_json(tmp_path):
    from visionservex.cli.model_health_commands import app

    out = tmp_path / "matrix.json"
    runner = CliRunner()
    result = runner.invoke(app, ["load-matrix", "--format", "json", "--out", str(out)])
    assert result.exit_code == 0
    payload = json.loads(out.read_text())
    assert payload["n_models"] > 0
    assert "rows" in payload


@pytest.mark.fast
def test_load_matrix_cli_writes_markdown(tmp_path):
    from visionservex.cli.model_health_commands import app

    out = tmp_path / "matrix.md"
    runner = CliRunner()
    result = runner.invoke(app, ["load-matrix", "--format", "markdown", "--out", str(out)])
    assert result.exit_code == 0
    body = out.read_text()
    assert "| Model | Family | Task | Mode" in body


# ---------------------------------------------------------------------------
# CLI help sweep
# ---------------------------------------------------------------------------


_PUBLIC_SUBAPPS = (
    "detect",
    "open-vocab",
    "segment",
    "classify",
    "embed",
    "similarity",
    "video-search",
    "anomaly",
    "medical",
    "openmmlab",
    "maskdino",
    "sam-family",
    "agriculture",
    "aerial",
    "benchmark-classification",
    "benchmark-anomaly",
    "benchmark-surveillance-search",
    "benchmark-open-vocab",
    "model-zoo",
    "models",
    "readiness",
    "florence2",
)


def _visionservex_bin() -> str:
    """Return the installed `visionservex` console script if available."""
    import shutil

    for candidate in ("visionservex",):
        path = shutil.which(candidate)
        if path:
            return path
    pytest.skip("`visionservex` console script not installed in this env")


@pytest.mark.fast
@pytest.mark.parametrize("subapp", _PUBLIC_SUBAPPS)
def test_cli_subapp_help_does_not_crash(subapp):
    binary = _visionservex_bin()
    result = subprocess.run(
        [binary, subapp, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # `--help` exits 0; some Typer versions exit 2 for groups. Either is fine.
    assert result.returncode in (0, 2), (
        f"`visionservex {subapp} --help` crashed (rc={result.returncode})\n"
        f"stderr: {result.stderr[:400]}"
    )
    assert "Traceback" not in (result.stdout or "")
    assert "Traceback" not in (result.stderr or "")
    assert "ModuleNotFoundError" not in (result.stderr or "")


@pytest.mark.fast
def test_visionservex_version_subcommand_works():
    binary = _visionservex_bin()
    result = subprocess.run(
        [binary, "version"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0
    assert "VisionServeX" in result.stdout


# ---------------------------------------------------------------------------
# Readiness gate
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_readiness_verdict_still_release_ok():
    from visionservex.cli.readiness_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["verdict", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["verdict"] == "RELEASE_OK"
    assert payload["all_ready"] is True
