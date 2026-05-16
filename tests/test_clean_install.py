# SPDX-License-Identifier: Apache-2.0
"""Clean-venv install validation.

Opt-in test that builds the current wheel into /tmp, installs it in a
brand-new venv, and confirms the installed `visionservex` console script
boots, runs `--help`, runs `readiness verdict`, and runs the new
`models load-matrix` command. Skipped by default because building the
wheel takes ~5 s — enable with VISIONSERVEX_RUN_CLEAN_INSTALL_TESTS=1.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)


@pytest.fixture(scope="module")
def clean_venv(tmp_path_factory):
    if not os.environ.get("VISIONSERVEX_RUN_CLEAN_INSTALL_TESTS"):
        pytest.skip("set VISIONSERVEX_RUN_CLEAN_INSTALL_TESTS=1 to enable")
    venv = tmp_path_factory.mktemp("vsx_clean_install")
    _run([sys.executable, "-m", "venv", str(venv)])
    pip = venv / "bin" / "pip"
    _run([str(pip), "install", "-q", "-U", "pip"], timeout=120)
    # Build a fresh wheel if no current dist file exists.
    dist = ROOT / "dist"
    wheel = next(dist.glob("visionservex-*-py3-none-any.whl"), None) if dist.exists() else None
    if wheel is None:
        _run([sys.executable, "-m", "build"], cwd=ROOT, timeout=600)
        wheel = next((ROOT / "dist").glob("visionservex-*-py3-none-any.whl"))
    _run([str(pip), "install", "-q", str(wheel)], timeout=600)
    return venv


@pytest.mark.real_model
def test_clean_venv_visionservex_version(clean_venv):
    binary = clean_venv / "bin" / "visionservex"
    assert binary.exists(), "console script not installed in venv"
    res = _run([str(binary), "version"], timeout=30)
    assert res.returncode == 0
    assert "VisionServeX" in res.stdout


@pytest.mark.real_model
def test_clean_venv_help_does_not_crash(clean_venv):
    binary = clean_venv / "bin" / "visionservex"
    res = _run([str(binary), "--help"], timeout=30)
    assert res.returncode == 0
    assert "Usage: visionservex" in res.stdout


@pytest.mark.real_model
def test_clean_venv_readiness_verdict_release_ok(clean_venv):
    binary = clean_venv / "bin" / "visionservex"
    res = _run([str(binary), "readiness", "verdict", "--json"], timeout=30)
    assert res.returncode == 0
    payload = json.loads(res.stdout)
    assert payload["verdict"] == "RELEASE_OK"


@pytest.mark.real_model
def test_clean_venv_load_matrix(clean_venv, tmp_path):
    binary = clean_venv / "bin" / "visionservex"
    out = tmp_path / "matrix.json"
    res = _run(
        [str(binary), "models", "load-matrix", "--format", "json", "--out", str(out)],
        timeout=60,
    )
    assert res.returncode == 0, res.stderr
    data = json.loads(out.read_text())
    assert data["n_models"] > 50
    assert {"core_load", "sidecar_validate"}.issubset(set(data["by_mode"]))
