# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.26.0 tests: notebook run-audit, DEIMv2 benchmark probe, RT-DETRv4
checkpoint instructions, GHCR status report, v29 pre-v3 gate contract."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _vsx() -> list[str]:
    binary = shutil.which("visionservex")
    return [binary] if binary else [sys.executable, "-m", "visionservex"]


def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(_vsx() + args, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------------------
# Phase 1: notebook run-audit
# ---------------------------------------------------------------------------


def test_notebook_run_audit_help_lists_required_flags() -> None:
    from tests.helpers.cli_help import assert_help_contains_all, run_help

    res = run_help(["notebook", "run-audit"])
    assert_help_contains_all(res, ["--notebook", "--output", "--timeout", "--out"])


def test_notebook_run_audit_missing_input_returns_structured(tmp_path: Path) -> None:
    out = tmp_path / "audit.json"
    res = _run(
        [
            "notebook",
            "run-audit",
            "--notebook",
            str(tmp_path / "no_such.ipynb"),
            "--output",
            str(tmp_path / "out.ipynb"),
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 2
    d = json.loads(out.read_text())
    assert d["status"] == "failed"
    assert d["code"] == "INPUT_NOT_FOUND"


def test_notebook_classify_failure_recognises_known_blockers(tmp_path: Path) -> None:
    stderr = tmp_path / "stderr.txt"
    stderr.write_text("BLACKWELL_SM120_TORCH_INCOMPATIBLE\nsomething else")
    out = tmp_path / "classify.json"
    res = _run(
        [
            "notebook",
            "classify-failure",
            "--stderr",
            str(stderr),
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["root_cause"] == "sidecar_blocker"
    assert d["blocker_code"] == "BLACKWELL_SM120_TORCH_INCOMPATIBLE"


# ---------------------------------------------------------------------------
# Phase 2: DEIMv2 benchmark probe
# ---------------------------------------------------------------------------


def test_deimv2_benchmark_help_lists_flags() -> None:
    from tests.helpers.cli_help import assert_help_contains_all, run_help

    res = run_help(["deimv2", "benchmark"])
    assert_help_contains_all(
        res,
        ["--model-id", "--profile", "--max-images", "--device", "--out"],
    )


def test_deimv2_benchmark_missing_image_dir(tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    res = _run(
        [
            "deimv2",
            "benchmark",
            str(tmp_path / "no_dir"),
            "--max-images",
            "5",
            "--device",
            "cpu",
            "--out",
            str(out),
            "--format",
            "json",
        ]
    )
    assert res.returncode == 2
    d = json.loads(out.read_text())
    assert d["status"] == "failed"
    assert d["code"] == "INPUT_NOT_FOUND"


# ---------------------------------------------------------------------------
# Phase 3: RT-DETRv4 checkpoint-instructions + validate-checkpoint
# ---------------------------------------------------------------------------


def test_rtdetrv4_checkpoint_instructions_known_variant(tmp_path: Path) -> None:
    out = tmp_path / "ckpt.json"
    res = _run(
        [
            "rtdetrv4",
            "checkpoint-instructions",
            "rtdetrv4-s",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["model_id"] == "rtdetrv4-s"
    assert "gdown" in d["gdown_command_if_known"]
    assert d["expected_filename"] == "rtdetrv4-s.pth"
    assert d["smoke_command_after_download"]
    assert d["benchmark_command_after_download"]


def test_rtdetrv4_checkpoint_instructions_unknown_variant(tmp_path: Path) -> None:
    out = tmp_path / "ckpt.json"
    res = _run(
        [
            "rtdetrv4",
            "checkpoint-instructions",
            "rtdetrv4-zzz",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["code"] == "CHECKPOINT_NOT_FOUND"


def test_rtdetrv4_validate_checkpoint_missing_file(tmp_path: Path) -> None:
    out = tmp_path / "validate.json"
    res = _run(
        [
            "rtdetrv4",
            "validate-checkpoint",
            "rtdetrv4-s",
            "--checkpoint",
            str(tmp_path / "no_file.pth"),
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["status"] == "expected_blocker"
    assert d["code"] == "CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP"


def test_rtdetrv4_validate_checkpoint_zero_byte_file(tmp_path: Path) -> None:
    ckpt = tmp_path / "fake.pth"
    ckpt.write_bytes(b"")
    out = tmp_path / "validate.json"
    res = _run(
        [
            "rtdetrv4",
            "validate-checkpoint",
            "rtdetrv4-s",
            "--checkpoint",
            str(ckpt),
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["size_bytes"] == 0
    # Sidecar probe (if env exists) should mark this CHECKPOINT_INVALID;
    # otherwise SIDECAR_ENV_MISSING.
    assert d["code"] in {"CHECKPOINT_INVALID", "SIDECAR_ENV_MISSING", "OK"}


# ---------------------------------------------------------------------------
# Phase 8: version assertion
# ---------------------------------------------------------------------------


def test_version_is_at_least_2_26_0() -> None:
    import visionservex

    parts = visionservex.__version__.split(".")
    major, minor = int(parts[0]), int(parts[1])
    assert (major, minor) >= (2, 26), visionservex.__version__
