# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.45.0: license opt-in CLI flags and env vars."""

from __future__ import annotations

import subprocess
import sys


def test_license_gate_check_agpl_model_blocked():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "visionservex.__main__",
            "license-gate",
            "check",
            "yolo26x.pt",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    # Should exit 1 (license gate not passed)
    assert proc.returncode == 1
    import json

    d = json.loads(proc.stdout)
    assert d["code"] == "LICENSE_GATE_NOT_PASSED"
    assert d["license"] == "AGPL-3.0"
    assert d["default_safe"] is False
    assert "VISIONSERVEX_ACCEPT_AGPL" in d["opt_in_env_var"]


def test_license_gate_check_agpl_model_allowed_with_flag():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "visionservex.__main__",
            "license-gate",
            "check",
            "yolo26x.pt",
            "--accept-agpl",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    import json

    d = json.loads(proc.stdout)
    assert d["code"] == "OPT_IN_ACCEPTED"


def test_license_gate_check_default_safe_model():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "visionservex.__main__",
            "license-gate",
            "check",
            "dfine-x",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    import json

    d = json.loads(proc.stdout)
    assert d["code"] == "DEFAULT_SAFE"
    assert d["default_safe"] is True


def test_registry_validate_deprecated_model():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "visionservex.__main__",
            "registry",
            "validate",
            "deim-m",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    import json

    d = json.loads(proc.stdout)
    assert d["final_state"] == "upstream_deprecated"
    assert d["can_benchmark"] is False


def test_registry_validate_wrong_entry():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "visionservex.__main__",
            "registry",
            "validate",
            "oneformer-convnext-large",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    import json

    d = json.loads(proc.stdout)
    assert d["final_state"] == "wrong_registry_entry"
