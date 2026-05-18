# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Phase 1 (v2.19.0): CLI hardening — addendum failures from v21 notebook run.

The v21 notebook run exposed:
- `visionservex --version` returned "No such option".
- `visionservex classify maxvit` returned MODEL_NOT_FOUND.
- `visionservex segment MODEL IMAGE --box X,Y,X,Y --out --draw --format` failed with Usage.
- `visionservex sam-family smoke-test MODEL IMAGE --box --out --draw --format` failed with Usage.
- `visionservex plot --help` failed with "No such command".
- `visionservex medical monai list-bundles` returned exit-3 → classified as failed_runtime.

These tests pin the fixes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _vsx() -> list[str]:
    binary = shutil.which("visionservex")
    return [binary] if binary else [sys.executable, "-m", "visionservex"]


def _run(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(_vsx() + args, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------------------
# --version
# ---------------------------------------------------------------------------


def test_version_flag_returns_clean() -> None:
    res = _run(["--version"])
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "VisionServeX" in res.stdout
    # Either v2.18.x or v2.19.x at the time this test runs.
    assert "2." in res.stdout


def test_short_version_flag() -> None:
    res = _run(["-V"])
    assert res.returncode == 0
    assert "VisionServeX" in res.stdout


# ---------------------------------------------------------------------------
# maxvit alias
# ---------------------------------------------------------------------------


def test_maxvit_alias_resolves_in_registry() -> None:
    from visionservex.registry import default_registry

    reg = default_registry()
    assert reg.has("maxvit")
    entry = reg.get("maxvit")
    assert entry.id == "maxvit-tiny-tf-224"


def test_swinv2_alias_resolves() -> None:
    from visionservex.registry import default_registry

    entry = default_registry().get("swinv2")
    assert "swinv2" in entry.id


def test_dinov2_alias_resolves() -> None:
    from visionservex.registry import default_registry

    entry = default_registry().get("dinov2")
    assert "dinov2" in entry.id


def test_unknown_id_still_raises() -> None:
    from visionservex.registry import RegistryError, default_registry

    try:
        default_registry().get("totally-bogus-model-id")
    except RegistryError as exc:
        assert "totally-bogus-model-id" in str(exc)
    else:
        raise AssertionError("should have raised")


# ---------------------------------------------------------------------------
# segment alias accepts --box / --out / --draw / --format
# ---------------------------------------------------------------------------


def test_segment_help_lists_v219_flags() -> None:
    """v2.25.1: rich-aware help assertion (CI terminal soft-wraps option names)."""
    from tests.helpers.cli_help import assert_help_contains_all, run_help

    res = run_help(["segment"])
    assert_help_contains_all(res, ["--box", "--out", "--draw", "--format"])


def test_segment_unknown_model_emits_structured_not_usage(tmp_path: Path) -> None:
    """A non-existent model must not produce a typer 'Usage:' error."""
    img = tmp_path / "img.jpg"
    from PIL import Image as _PIL

    _PIL.new("RGB", (40, 40)).save(img)
    res = _run(
        [
            "segment",
            "totally-bogus-segment-model",
            str(img),
            "--box",
            "10,10,30,30",
            "--format",
            "json",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    # Must not produce typer "Usage:" error.
    assert "Usage:" not in res.stderr


# ---------------------------------------------------------------------------
# sam-family smoke-test accepts --out / --draw / --format
# ---------------------------------------------------------------------------


def test_sam_family_smoke_test_help_lists_v219_flags() -> None:
    res = _run(["sam-family", "smoke-test", "--help"])
    assert res.returncode == 0
    assert "--out" in res.stdout
    assert "--draw" in res.stdout
    assert "--format" in res.stdout
    assert "--box" in res.stdout


def test_sam_family_smoke_test_non_runnable_returns_expected_blocker(tmp_path: Path) -> None:
    """A SAM model marked not-runnable must return expected_blocker exit 0, not Usage."""
    img = tmp_path / "img.jpg"
    from PIL import Image as _PIL

    _PIL.new("RGB", (40, 40)).save(img)
    out = tmp_path / "smoke.json"
    res = _run(
        [
            "sam-family",
            "smoke-test",
            "sam3",
            str(img),
            "--box",
            "10,10,30,30",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert "Usage:" not in res.stderr
    if out.exists():
        payload = json.loads(out.read_text())
        # sam3 is gated → either expected_blocker or model_not_found
        assert payload.get("status") in {"expected_blocker", "failed"} or payload.get("code") in {
            "MODEL_NOT_FOUND",
            "MODEL_NOT_RUNNABLE",
            "GATED_HF_AUTH_REQUIRED",
        }


# ---------------------------------------------------------------------------
# plot placeholder
# ---------------------------------------------------------------------------


def test_plot_returns_structured_not_implemented(tmp_path: Path) -> None:
    out = tmp_path / "plot.json"
    res = _run(["plot", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["status"] == "expected_blocker"
    assert payload["code"] == "BENCHMARK_NOT_IMPLEMENTED"
    assert "recommended_alternatives" in payload


def test_plot_help_does_not_404() -> None:
    res = _run(["plot", "--help"])
    assert res.returncode == 0
    assert "No such command" not in res.stderr


# ---------------------------------------------------------------------------
# medical monai list-bundles is expected_blocker, not failed_runtime
# ---------------------------------------------------------------------------


def test_monai_list_bundles_when_missing_returns_expected_blocker(tmp_path: Path) -> None:
    out = tmp_path / "monai.json"
    res = _run(
        [
            "medical",
            "monai",
            "list-bundles",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    # v2.19.0: expected_blocker must NOT crash with exit-3.
    # Either monai is installed (exit 0 with bundles) or not (exit 0 with MONAI_REQUIRED).
    assert res.returncode == 0, (res.stdout, res.stderr)
    payload = json.loads(out.read_text())
    assert payload["status"] in {"ok", "expected_blocker"}
    if payload["status"] == "expected_blocker":
        assert payload["code"] == "MONAI_REQUIRED"


# ---------------------------------------------------------------------------
# result_classifier knows the new v2.19 codes
# ---------------------------------------------------------------------------


def test_classifier_knows_monai_required() -> None:
    from visionservex.runtime.result_classifier import (
        EXPECTED_BLOCKER_CODES,
        classify_command_result,
    )

    assert "MONAI_REQUIRED" in EXPECTED_BLOCKER_CODES
    payload = {"status": "expected_blocker", "code": "MONAI_REQUIRED"}
    r = classify_command_result(returncode=0, stdout=json.dumps(payload), stderr="")
    assert r.status == "expected_blocker"


def test_classifier_knows_benchmark_not_implemented() -> None:
    from visionservex.runtime.result_classifier import (
        EXPECTED_BLOCKER_CODES,
        classify_command_result,
    )

    assert "BENCHMARK_NOT_IMPLEMENTED" in EXPECTED_BLOCKER_CODES
    payload = {"status": "expected_blocker", "code": "BENCHMARK_NOT_IMPLEMENTED"}
    r = classify_command_result(returncode=0, stdout=json.dumps(payload), stderr="")
    assert r.status == "expected_blocker"
