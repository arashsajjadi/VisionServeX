# SPDX-License-Identifier: Apache-2.0
"""v3.8 — `visionservex model pull` enforces the license policy before downloading."""

from __future__ import annotations

from typer.testing import CliRunner

from visionservex.cli.main import app

runner = CliRunner()


def test_byot_pull_refused_without_accept_flag():
    res = runner.invoke(app, ["model", "pull", "sam3-base", "--json"])
    assert res.exit_code == 2
    assert "UPSTREAM_LICENSE_NOT_ACCEPTED" in res.output


def test_noncommercial_pull_refused_without_flags():
    res = runner.invoke(app, ["model", "pull", "edge-sam", "--json"])
    assert res.exit_code == 2
    assert "NONCOMMERCIAL_REFUSED" in res.output


def test_enterprise_pull_refused():
    res = runner.invoke(app, ["model", "pull", "fastsam-s", "--json"])
    assert res.exit_code == 2
    assert "ENTERPRISE_LICENSE_REQUIRED" in res.output


def test_legal_review_pull_refused():
    res = runner.invoke(app, ["model", "pull", "tinysam", "--json"])
    assert res.exit_code == 2
    assert "LEGAL_REVIEW_REQUIRED" in res.output


def test_external_api_pull_refused():
    res = runner.invoke(app, ["model", "pull", "dino-x-api", "--json"])
    assert res.exit_code == 2
    assert "EXTERNAL_API_ONLY_TERMS_REQUIRED" in res.output


def test_model_license_command_outputs_policy():
    res = runner.invoke(app, ["model", "license", "sam3-base", "--json"])
    assert res.exit_code == 0
    assert "byot_license_required" in res.output
    assert "can_ship_weights" in res.output


def test_hf_token_env_must_be_set():
    res = runner.invoke(app, ["model", "pull", "sam3-base",
                              "--hf-token-env", "DEFINITELY_NOT_SET_VAR", "--json"])
    assert res.exit_code != 0
    assert "NO_TOKEN_IN_ENV" in res.output
