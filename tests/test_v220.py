# SPDX-License-Identifier: Apache-2.0
"""Tests for v2.2.0: benchmark CLIs, Florence-2 create-env, SAM2.1 manifest/registry."""

from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------------------
# benchmark-classification CLI registration
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_benchmark_classification_registered():
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["benchmark-classification", "--help"])
    assert result.exit_code == 0
    assert "classification" in result.output.lower()


# ---------------------------------------------------------------------------
# benchmark-anomaly CLI registration
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_benchmark_anomaly_registered():
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["benchmark-anomaly", "--help"])
    assert result.exit_code == 0
    assert "anomaly" in result.output.lower()


# ---------------------------------------------------------------------------
# benchmark-surveillance-search CLI registration
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_benchmark_surveillance_registered():
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["benchmark-surveillance-search", "--help"])
    assert result.exit_code == 0
    assert "surveillance" in result.output.lower()


# ---------------------------------------------------------------------------
# florence2 create-env command
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_florence2_create_env_json_output():
    from typer.testing import CliRunner

    from visionservex.cli.florence2_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["create-env", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert "env_name" in payload
    assert "commands" in payload
    assert len(payload["commands"]) >= 4
    assert any("transformers>=4.40,<5.0" in cmd for cmd in payload["commands"])


@pytest.mark.fast
def test_florence2_create_env_custom_name():
    from typer.testing import CliRunner

    from visionservex.cli.florence2_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["create-env", "--name", "my-env", "--python", "3.10", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip())
    assert payload["env_name"] == "my-env"
    assert payload["python"] == "3.10"
    assert any("my-env" in cmd for cmd in payload["commands"])


@pytest.mark.fast
def test_florence2_create_env_no_execute_by_default():
    """create-env without --execute must NOT run any subprocess."""
    from unittest.mock import patch

    from typer.testing import CliRunner

    from visionservex.cli.florence2_commands import app

    runner = CliRunner()
    with patch("subprocess.run") as mock_run:
        result = runner.invoke(app, ["create-env", "--name", "test-env"])
        mock_run.assert_not_called()
    assert result.exit_code == 0


@pytest.mark.fast
def test_florence2_create_env_execute_no_conda(tmp_path):
    """--execute with no conda in PATH returns exit code 2."""
    from unittest.mock import patch

    from typer.testing import CliRunner

    from visionservex.cli.florence2_commands import app

    runner = CliRunner()
    with patch("shutil.which", return_value=None):
        result = runner.invoke(app, ["create-env", "--execute"])
    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# SAM2.1 manifest entries — all 4 variants present, no duplicate keys
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_sam21_all_four_in_manifest():
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST as MODEL_SOURCES

    for mid in (
        "sam2.1-hiera-tiny",
        "sam2.1-hiera-small",
        "sam2.1-hiera-base-plus",
        "sam2.1-hiera-large",
    ):
        assert mid in MODEL_SOURCES, f"Missing from manifest: {mid}"


@pytest.mark.fast
def test_sam21_hf_repos_correct():
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST as MODEL_SOURCES

    for suffix in ("tiny", "small", "base-plus", "large"):
        mid = f"sam2.1-hiera-{suffix}"
        src = MODEL_SOURCES[mid]
        assert src.hf_repo == f"facebook/sam2.1-hiera-{suffix}", (
            f"{mid} has wrong hf_repo: {src.hf_repo}"
        )


@pytest.mark.fast
def test_sam21_license_apache():
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST as MODEL_SOURCES

    for suffix in ("tiny", "small", "base-plus", "large"):
        mid = f"sam2.1-hiera-{suffix}"
        src = MODEL_SOURCES[mid]
        assert "Apache" in src.license, f"{mid} license unexpected: {src.license}"


@pytest.mark.fast
def test_sam21_no_duplicate_in_manifest():
    """Ensure the old audit_only sam2.1-hiera-large duplicate was removed."""
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST as MODEL_SOURCES

    # Each key must appear exactly once (dict guarantees this by design,
    # but we check that the surviving entry is the correct add_now one)
    src = MODEL_SOURCES["sam2.1-hiera-large"]
    assert src.recommended_action == "add_now", f"Expected add_now, got: {src.recommended_action}"


# ---------------------------------------------------------------------------
# SAM2.1 runtime registry — entries present
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_sam21_in_runtime_registry():
    from visionservex.registry import default_registry

    reg = default_registry()
    for mid in (
        "sam2.1-hiera-tiny",
        "sam2.1-hiera-small",
        "sam2.1-hiera-base-plus",
        "sam2.1-hiera-large",
    ):
        entry = reg.get(mid)
        assert entry is not None, f"Missing from runtime registry: {mid}"
        assert entry.hf_repo_id == f"facebook/sam2.1-hiera-{mid[len('sam2.1-hiera-') :]}"


# ---------------------------------------------------------------------------
# Lightweight SAM — license decisions recorded in manifest
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_fastsam_excluded_agpl():
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST as MODEL_SOURCES

    for mid in ("fastsam-s", "fastsam-x"):
        assert mid in MODEL_SOURCES
        src = MODEL_SOURCES[mid]
        assert src.license == "AGPL-3.0"
        assert src.recommended_action == "do_not_add"
        assert src.runnable_in_visionservex is False


@pytest.mark.fast
def test_mobilesam_efficientsam_expert_sidecar():
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST as MODEL_SOURCES

    for mid in ("mobilesam", "efficientsam", "hq-sam", "edgesam"):
        assert mid in MODEL_SOURCES, f"Missing: {mid}"
        src = MODEL_SOURCES[mid]
        assert src.license == "Apache-2.0", f"{mid} unexpected license: {src.license}"
        assert src.recommended_action == "expert_sidecar"
        assert src.runnable_in_visionservex is False
