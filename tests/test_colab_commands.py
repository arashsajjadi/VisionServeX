# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tests for the Colab GPU worker CLI subgroup."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from visionservex.cli.main import app

runner = CliRunner()


# ---------- detection helpers ----------


def test_in_colab_returns_false_off_colab():
    from visionservex.cli.colab_commands import in_colab

    assert in_colab() is False


def test_colab_doctor_not_in_colab_json():
    """Outside Colab, doctor returns COLAB_NOT_DETECTED."""
    r = runner.invoke(app, ["colab", "doctor", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["in_colab"] is False
    assert data["status"] == "COLAB_NOT_DETECTED"
    assert "hint" in data


def test_colab_status_off_colab():
    r = runner.invoke(app, ["colab", "status", "--json"])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert data["in_colab"] is False


def test_colab_doctor_simulated_colab_with_gpu():
    """Simulate Colab with a healthy GPU."""
    with (
        patch("visionservex.cli.colab_commands.in_colab", return_value=True),
        patch(
            "visionservex.cli.colab_commands._detect_gpu",
            return_value={
                "available": True,
                "name": "Tesla T4",
                "total_vram_gb": 14.7,
                "free_vram_gb": 14.0,
            },
        ),
    ):
        r = runner.invoke(app, ["colab", "doctor", "--json"])
        assert r.exit_code == 0, r.output
        data = json.loads(r.output)
        assert data["in_colab"] is True
        assert data["status"] == "ok"
        assert data["gpu"]["available"] is True
        assert data["gpu"]["name"] == "Tesla T4"
        assert data["safe_vram_budget_gb"] == pytest.approx(12.5, abs=0.1)


def test_colab_doctor_simulated_colab_no_gpu():
    """Simulate Colab without GPU."""
    with (
        patch("visionservex.cli.colab_commands.in_colab", return_value=True),
        patch(
            "visionservex.cli.colab_commands._detect_gpu",
            return_value={
                "available": False,
                "name": None,
                "total_vram_gb": None,
                "free_vram_gb": None,
            },
        ),
    ):
        r = runner.invoke(app, ["colab", "doctor", "--json"])
        assert r.exit_code == 0
        data = json.loads(r.output)
        assert data["status"] == "COLAB_GPU_UNAVAILABLE"
        assert "Runtime" in data["hint"]


def test_colab_gpu_check_no_gpu_exits_nonzero():
    with patch(
        "visionservex.cli.colab_commands._detect_gpu",
        return_value={"available": False, "name": None},
    ):
        r = runner.invoke(app, ["colab", "gpu-check", "--json"])
        assert r.exit_code == 1
        data = json.loads(r.output)
        assert data["status"] == "COLAB_GPU_UNAVAILABLE"


def test_colab_gpu_check_with_gpu():
    with patch(
        "visionservex.cli.colab_commands._detect_gpu",
        return_value={
            "available": True,
            "name": "Tesla T4",
            "total_vram_gb": 14.7,
            "free_vram_gb": 14.0,
        },
    ):
        r = runner.invoke(app, ["colab", "gpu-check", "--json"])
        assert r.exit_code == 0
        data = json.loads(r.output)
        assert data["status"] == "ok"
        assert data["recommended_max_concurrency"] == 1
        assert data["recommended_profile"] == "colab-gpu-worker"


# ---------- cache and drive ----------


def test_colab_cache_path_ephemeral():
    """Without Drive mounted, cache path is ephemeral with warning."""
    with patch("visionservex.cli.colab_commands._drive_mounted", return_value=False):
        r = runner.invoke(app, ["colab", "cache-path", "--json"])
        assert r.exit_code == 0
        data = json.loads(r.output)
        assert data["persistent"] is False
        assert data["cache_dir"] == "/content/visionservex_cache"
        assert data["warning"] is not None


def test_colab_cache_path_drive():
    with patch("visionservex.cli.colab_commands._drive_mounted", return_value=True):
        r = runner.invoke(app, ["colab", "cache-path", "--json"])
        assert r.exit_code == 0
        data = json.loads(r.output)
        assert data["persistent"] is True
        assert "drive" in data["cache_dir"].lower()


def test_colab_setup_cache_drive_not_mounted_fails():
    with patch("visionservex.cli.colab_commands._drive_mounted", return_value=False):
        r = runner.invoke(app, ["colab", "setup-cache", "--drive", "--json"])
        assert r.exit_code == 1
        data = json.loads(r.output)
        assert data["status"] == "DRIVE_NOT_MOUNTED"


def test_colab_setup_cache_local():
    r = runner.invoke(app, ["colab", "setup-cache", "--json"])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert data["cache_dir"] == "/content/visionservex_cache"
    assert "VISIONSERVEX_CACHE_DIR" in data["env_command"]


# ---------- token ----------


def test_colab_token_returns_url_safe_string():
    r = runner.invoke(app, ["colab", "token", "--json"])
    assert r.exit_code == 0
    data = json.loads(r.output)
    assert "api_key" in data
    assert len(data["api_key"]) >= 32
    assert "VISIONSERVEX_AUTH__API_KEY" in data["env_command"]


# ---------- tunnel safety ----------


def test_colab_tunnel_start_refuses_without_auth():
    with patch("visionservex.cli.colab_commands._detect_auth_configured", return_value=False):
        r = runner.invoke(
            app,
            [
                "colab",
                "tunnel-start",
                "--domain",
                "test.example.com",
                "--i-understand-this-is-public",
                "--json",
            ],
        )
        assert r.exit_code == 1
        data = json.loads(r.output)
        assert data["status"] == "AUTH_REQUIRED"


def test_colab_tunnel_start_refuses_without_acknowledgement():
    with patch("visionservex.cli.colab_commands._detect_auth_configured", return_value=True):
        r = runner.invoke(
            app,
            ["colab", "tunnel-start", "--domain", "test.example.com", "--json"],
        )
        assert r.exit_code == 1
        data = json.loads(r.output)
        assert data["status"] == "EXPOSURE_NOT_ACKNOWLEDGED"


def test_colab_tunnel_start_refuses_without_cloudflared():
    with (
        patch("visionservex.cli.colab_commands._detect_auth_configured", return_value=True),
        patch("visionservex.cli.colab_commands.shutil.which", return_value=None),
    ):
        r = runner.invoke(
            app,
            [
                "colab",
                "tunnel-start",
                "--domain",
                "test.example.com",
                "--i-understand-this-is-public",
                "--json",
            ],
        )
        assert r.exit_code == 1
        data = json.loads(r.output)
        assert data["status"] == "CLOUDFLARED_MISSING"


# ---------- remote client ----------


def test_colab_test_remote_unreachable():
    """If remote URL is unreachable, return UNREACHABLE."""
    r = runner.invoke(
        app,
        [
            "colab",
            "test-remote",
            "http://127.0.0.1:1",  # closed port — should fail
            "--timeout",
            "1",
            "--json",
        ],
    )
    assert r.exit_code == 1
    data = json.loads(r.output)
    assert data["status"] == "UNREACHABLE"


# ---------- profile ----------


def test_colab_gpu_worker_profile_exists():
    """Confirm the colab-gpu-worker profile is registered."""
    from visionservex.cli.gateway_commands import _PROFILES

    assert "colab-gpu-worker" in _PROFILES
    profile = _PROFILES["colab-gpu-worker"]
    # Must bind to localhost
    assert profile["VISIONSERVEX_SERVER__HOST"] == "127.0.0.1"
    # Must have conservative concurrency
    assert profile["VISIONSERVEX_RUNTIME__MAX_LOADED_MODELS"] == "1"
    assert profile["VISIONSERVEX_RUNTIME__PER_MODEL_CONCURRENCY"] == "1"
    # Must not auto-save inputs/outputs
    assert profile["VISIONSERVEX_PRIVACY__SAVE_INPUTS"] == "false"
    assert profile["VISIONSERVEX_PRIVACY__SAVE_OUTPUTS"] == "false"


def test_colab_gateway_profile_listed():
    """The colab-gpu-worker profile is discoverable via gateway profile-list."""
    r = runner.invoke(app, ["gateway", "profile-list", "--json"])
    assert r.exit_code == 0
    data = json.loads(r.output)
    # profile-list returns a list of profile names
    assert "colab-gpu-worker" in data


def test_colab_help_lists_commands():
    """`visionservex colab --help` lists all colab subcommands."""
    r = runner.invoke(app, ["colab", "--help"])
    assert r.exit_code == 0
    for cmd in ("doctor", "status", "gpu-check", "cache-path", "setup-cache", "token"):
        assert cmd in r.output, f"missing command: {cmd}"
