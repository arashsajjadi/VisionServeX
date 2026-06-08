# SPDX-License-Identifier: Apache-2.0
"""v3.11: INSID3 CLI commands — import, structure, status command smoke test."""

from __future__ import annotations


def test_insid3_commands_importable():
    from visionservex.cli import insid3_commands

    assert hasattr(insid3_commands, "app")


def test_insid3_app_has_expected_commands():
    from typer.testing import CliRunner

    from visionservex.cli.insid3_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    help_text = result.output
    assert "status" in help_text
    assert "doctor" in help_text
    assert "run" in help_text
    assert "correspond" in help_text


def test_insid3_status_command_runs():
    from typer.testing import CliRunner

    from visionservex.cli.insid3_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "insid3" in result.output.lower()
    assert "facebook/dinov3" in result.output


def test_insid3_in_main_cli():
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["insid3", "--help"])
    assert result.exit_code == 0
    assert "insid3" in result.output.lower() or "INSID3" in result.output


def test_insid3_run_help():
    from typer.testing import CliRunner

    from visionservex.cli.insid3_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "query-image" in result.output or "query_image" in result.output.lower()
