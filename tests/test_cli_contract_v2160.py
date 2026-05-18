# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Phase 6 (v2.16.0): CLI contract regressions caught by the v16 notebook run.

The notebook used commands like::

    visionservex classify swinv2-tiny img.jpg --top-k 5 --out X.json --format json
    visionservex similarity siglip2-base-patch16-224 a.jpg b.jpg --out X.json --format json
    visionservex agriculture model-card agriclip --format json --out X.json
    visionservex video-search tracker-smoke --tracker bytetrack ... --format json
    visionservex video-search reid-smoke --reid osnet ... --format json
    visionservex benchmark-anomaly --dataset simple:... --model patchcore ... --format json

These produced "Usage:" errors because the corresponding subcommands were
missing ``--out``, ``--format``, or (for agriculture) the entire ``model-card``
subcommand. This test asserts every one of those commands at least parses
its arguments cleanly.

We don't run inference here (that would download multi-GB checkpoints);
we only verify that argument parsing succeeds and that --help is sane.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _vsx_cmd() -> list[str]:
    binary = shutil.which("visionservex")
    return [binary] if binary else [sys.executable, "-m", "visionservex"]


def _run(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(_vsx_cmd() + args, capture_output=True, text=True, timeout=timeout)


def test_classify_alias_help_lists_out_and_format() -> None:
    """`visionservex classify --help` must mention --out and --format (notebook contract).

    v2.25.0: use rich-aware help assertion so CI's narrower terminal width
    can't soft-wrap option names across lines and break substring matches.
    """
    from tests.helpers.cli_help import assert_help_contains_all, run_help

    res = run_help(["classify"])
    assert_help_contains_all(res, ["--out", "--format", "--top-k"])


def test_similarity_alias_help_lists_out_and_format() -> None:
    from tests.helpers.cli_help import assert_help_contains_all, run_help

    res = run_help(["similarity"])
    assert_help_contains_all(res, ["--out", "--format"])


def test_agriculture_model_card_help_exists() -> None:
    """Agriculture must expose `model-card` (it did not in v2.15)."""
    from tests.helpers.cli_help import assert_help_contains_all, run_help

    res = run_help(["agriculture", "model-card"])
    assert_help_contains_all(res, ["--out", "--format"])


def test_agriculture_model_card_agriclip_returns_expected_blocker(tmp_path: Path) -> None:
    out = tmp_path / "agriclip.json"
    res = _run(["agriculture", "model-card", "agriclip", "--format", "json", "--out", str(out)])
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["status"] == "expected_blocker"
    assert payload["code"] == "SIDECAR_REQUIRED"
    assert payload["model_id"] == "agriclip"
    assert payload["known_blockers"]


def test_agriculture_model_card_scold_returns_expected_blocker(tmp_path: Path) -> None:
    out = tmp_path / "scold.json"
    res = _run(["agriculture", "model-card", "scold", "--format", "json", "--out", str(out)])
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["status"] == "expected_blocker"
    assert payload["code"] == "SIDECAR_REQUIRED"
    assert payload["model_id"] == "scold"


def test_video_search_tracker_smoke_help_lists_format() -> None:
    from tests.helpers.cli_help import assert_help_contains_all, run_help

    res = run_help(["video-search", "tracker-smoke"])
    assert_help_contains_all(res, ["--format", "--out"])


def test_video_search_reid_smoke_help_lists_format() -> None:
    from tests.helpers.cli_help import assert_help_contains_all, run_help

    res = run_help(["video-search", "reid-smoke"])
    assert_help_contains_all(res, ["--format", "--out"])


def test_benchmark_anomaly_help_lists_format() -> None:
    from tests.helpers.cli_help import assert_help_contains_all, run_help

    res = run_help(["benchmark-anomaly"])
    assert_help_contains_all(res, ["--format", "--out"])


def test_dev_gpu_profile_help_lists_format() -> None:
    from tests.helpers.cli_help import assert_help_contains_all, run_help

    res = run_help(["dev", "gpu-profile"])
    assert_help_contains_all(res, ["--format", "--out"])


def test_sam_family_validate_sam31_returns_expected_blocker(tmp_path: Path) -> None:
    """sam-family validate sam3.1 must be `expected_blocker` (gated auth), not `fail`."""
    out = tmp_path / "sam31.json"
    res = _run(["sam-family", "validate", "sam3.1", "--format", "json", "--out", str(out)])
    # exit-code-0 with status=expected_blocker is the contract.
    assert out.exists(), (res.stdout, res.stderr)
    payload = json.loads(out.read_text())
    assert payload["status"] == "expected_blocker", payload
    assert payload["code"] == "GATED_HF_AUTH_REQUIRED", payload
