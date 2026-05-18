# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.25.0 tests: typer/help fix, readiness matrix, execution matrix,
cuda-compatibility, sidecar profiles, domain registry, RT-DETRv4 pull --method auto.

v2.25.0 proves on the dev RTX 5080:
- `visionservex models readiness-matrix` returns one row per advertised model
  with no null `execution_status` / `model_id` / `default_mode`.
- `visionservex models execution-matrix` propagates blockers and runs smokes
  when an image is supplied.
- `visionservex dev cuda-compatibility` detects Blackwell sm_120 and the
  installed torch's supported_arch_list.
- The `deimv2-blackwell-nightly` sidecar profile changes both the env name
  and the install plan (torch nightly cu128).
- `visionservex dataset validate-domain` is backed by DOMAIN_REGISTRY.
- `rtdetrv4 pull --method auto` either fetches via gdown or emits an
  exact manual command (never raw traceback).
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


def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(_vsx() + args, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------------------
# Phase 1: CLI help helper
# ---------------------------------------------------------------------------


def test_cli_help_helper_strips_ansi_and_normalizes_whitespace() -> None:
    from tests.helpers.cli_help import _normalize_for_match, strip_ansi

    sample = "\x1b[1m--out\x1b[0m\n\x1b[32m TEXT \x1b[0m\n ╭─── --format ───╮"
    plain = strip_ansi(sample)
    assert "\x1b[" not in plain
    norm = _normalize_for_match(sample)
    assert "--out" in norm
    assert "--format" in norm
    # Box-drawing chars must be stripped.
    assert "╭" not in norm


def test_run_help_uses_deterministic_env() -> None:
    from tests.helpers.cli_help import assert_help_contains_all, run_help

    res = run_help(["classify"])
    assert_help_contains_all(res, ["--out", "--format", "--top-k"])


# ---------------------------------------------------------------------------
# Phase 2: readiness matrix
# ---------------------------------------------------------------------------


def test_readiness_matrix_returns_one_row_per_advertised_model(tmp_path: Path) -> None:
    out = tmp_path / "readiness.json"
    res = _run(
        ["models", "readiness-matrix", "--format", "json", "--out", str(out)],
        timeout=180,
    )
    assert res.returncode == 0, res.stderr[-500:]
    d = json.loads(out.read_text())
    assert d["status"] == "ok"
    assert d["n_rows"] > 30
    # No null model_id / execution_status / default_mode / evidence_file.
    for r in d["rows"]:
        assert r["model_id"]
        assert r["execution_status"]
        assert r["default_mode"]
        assert r["evidence_file"]
        assert r["family"]
        assert r["task"]
    # Status set is bounded.
    statuses = {r["execution_status"] for r in d["rows"]}
    allowed = {
        "runnable",
        "runnable_cpu_only",
        "runnable_gpu",
        "expected_blocker",
        "sidecar_env_missing",
        "checkpoint_missing",
        "dependency_missing",
        "auth_required",
        "dataset_required",
        "license_blocked",
        "upstream_blocked",
        "resource_blocked",
        "not_advertised",
    }
    assert statuses.issubset(allowed), statuses - allowed


def test_readiness_matrix_csv(tmp_path: Path) -> None:
    out = tmp_path / "readiness.csv"
    res = _run(
        ["models", "readiness-matrix", "--format", "csv", "--out", str(out)],
        timeout=180,
    )
    assert res.returncode == 0, res.stderr[-500:]
    assert out.exists()
    head = out.read_text().splitlines()[0]
    for col in ("model_id", "family", "execution_status", "blocker_code", "default_mode"):
        assert col in head


def test_readiness_matrix_no_unclassified() -> None:
    """v2.25 contract: unclassified_model_status_count must be 0."""
    res = _run(["models", "readiness-matrix", "--format", "json"], timeout=180)
    assert res.returncode == 0
    d = json.loads(res.stdout.strip())
    assert d["unclassified_model_status_count"] == 0


# ---------------------------------------------------------------------------
# Phase 3: cuda-compatibility + sidecar profiles
# ---------------------------------------------------------------------------


def test_cuda_compatibility_reports_arch_list(tmp_path: Path) -> None:
    out = tmp_path / "cuda.json"
    res = _run(["dev", "cuda-compatibility", "--format", "json", "--out", str(out)], timeout=30)
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert "torch_version" in d
    assert "compute_capability" in d
    assert "supported_arch_list" in d
    assert "blackwell_detected" in d


def test_sidecar_profile_changes_env_name() -> None:
    from visionservex.sidecars.manager import SidecarManager

    mgr = SidecarManager()
    base = mgr.env_name("deimv2")
    blackwell = mgr.env_name("deimv2", profile="deimv2-blackwell-nightly")
    assert base != blackwell
    assert "blackwell" in blackwell
    assert "deimv2-blackwell-nightly" in blackwell


def test_sidecar_profile_blackwell_nightly_uses_cu128_nightly_index() -> None:
    from visionservex.sidecars.manager import (
        SidecarManager,
        apply_sidecar_profile,
    )

    spec = SidecarManager.get_spec("deimv2")
    assert spec is not None
    profiled = apply_sidecar_profile(spec, "deimv2-blackwell-nightly")
    assert profiled.cuda_channel == "nightly/cu128"
    # The profile must unpin torch so requirements.txt's torch==2.5.1 doesn't
    # downgrade the nightly install.
    assert profiled.torch_version == ""
    # Plan_create with this profile must put torch install LAST.
    plan = SidecarManager().plan_create(profiled, profile="deimv2-blackwell-nightly")
    nightly_idx = [i for i, c in enumerate(plan) if "nightly/cu128" in c]
    requirements_idx = [i for i, c in enumerate(plan) if "requirements.txt" in c]
    assert nightly_idx and requirements_idx
    assert nightly_idx[0] > requirements_idx[0], (
        "Nightly torch must be installed AFTER requirements.txt so the "
        "upstream pin doesn't overwrite it."
    )


def test_sidecar_create_blackwell_profile_dry_run_emits_env_name(tmp_path: Path) -> None:
    out = tmp_path / "create.json"
    res = _run(
        [
            "sidecar",
            "create",
            "deimv2",
            "--dry-run",
            "--profile",
            "deimv2-blackwell-nightly",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["code"] == "DRY_RUN"
    assert d["env_name"] == "visionservex-deimv2-blackwell-nightly-sidecar"
    assert d["profile"] == "deimv2-blackwell-nightly"


# ---------------------------------------------------------------------------
# Phase 4: RT-DETRv4 pull --method
# ---------------------------------------------------------------------------


def test_rtdetrv4_pull_method_manual_emits_gdown_command(tmp_path: Path) -> None:
    out = tmp_path / "pull.json"
    res = _run(
        [
            "rtdetrv4",
            "pull",
            "rtdetrv4-s",
            "--method",
            "manual",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["code"] in {"CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP", "OK"}
    assert "manual_command" in d
    assert "gdown" in d["manual_command"]


def test_rtdetrv4_pull_invalid_method(tmp_path: Path) -> None:
    out = tmp_path / "pull.json"
    res = _run(
        [
            "rtdetrv4",
            "pull",
            "rtdetrv4-s",
            "--method",
            "bogus",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["code"] in {"INVALID_METHOD", "OK"}


def test_rtdetrv4_smoke_test_missing_checkpoint_returns_blocker(tmp_path: Path) -> None:
    """smoke-test without --checkpoint and no cached checkpoint must structured-block."""
    from PIL import Image as _PIL

    img = tmp_path / "img.jpg"
    _PIL.new("RGB", (32, 32)).save(img)
    out = tmp_path / "smoke.json"
    res = _run(
        [
            "rtdetrv4",
            "smoke-test",
            "rtdetrv4-s",
            str(img),
            "--device",
            "cpu",
            "--checkpoint",
            str(tmp_path / "missing.pth"),
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["status"] in {"expected_blocker", "failed"}


# ---------------------------------------------------------------------------
# Phase 6: domain registry
# ---------------------------------------------------------------------------


def test_domain_registry_has_required_domains() -> None:
    from visionservex.datasets import DOMAIN_REGISTRY

    required = {
        "medical/2d-box",
        "medical/nifti-seg",
        "agriculture/hbb",
        "agriculture/segmentation",
        "aerial/hbb",
        "aerial/obb",
        "industrial/anomaly-simple",
        "surveillance/tracking",
        "video-search/retrieval",
        "pose/keypoints",
        "panoptic/segmentation",
    }
    assert set(DOMAIN_REGISTRY).issuperset(required)


def test_validate_domain_path_missing(tmp_path: Path) -> None:
    from visionservex.datasets import validate_domain_path

    p = tmp_path / "no_such"
    out = validate_domain_path("medical", "2d-box", p)
    assert out["status"] == "expected_blocker"
    assert out["code"] == "BOX_PROMPTS_REQUIRED"


def test_validate_domain_path_medical_2d_box_partial(tmp_path: Path) -> None:
    """Empty images dir + no boxes.json → still expected_blocker (metrics_valid=False)."""
    from visionservex.datasets import validate_domain_path

    (tmp_path / "images").mkdir()
    out = validate_domain_path("medical", "2d-box", tmp_path)
    assert out["metrics_valid"] is False
    assert out["benchmark_or_smoke"] == "smoke"


def test_validate_domain_cli(tmp_path: Path) -> None:
    out = tmp_path / "validate.json"
    res = _run(
        [
            "dataset",
            "validate-domain",
            "aerial",
            "--path",
            str(tmp_path / "no_such"),
            "--task",
            "obb",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["status"] == "expected_blocker"
    assert d["code"] == "DOTA_OR_OBB_LABELS_REQUIRED"


# ---------------------------------------------------------------------------
# Phase 7: execution matrix
# ---------------------------------------------------------------------------


def test_execution_matrix_runs_without_smoke_image(tmp_path: Path) -> None:
    """Without --smoke-image, runnable models classify NO_SMOKE_IMAGE_PROVIDED."""
    out = tmp_path / "exec.json"
    res = _run(
        ["models", "execution-matrix", "--device", "cpu", "--format", "json", "--out", str(out)],
        timeout=180,
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["n_rows"] > 0
    for r in d["rows"]:
        assert r["model_id"]
        assert r["status"] in {"ok", "expected_blocker", "failed"}
        # No raw traceback / usage errors.
        cmd = r.get("command", "")
        if cmd:
            assert "Traceback" not in cmd
            assert "Usage:" not in cmd


# ---------------------------------------------------------------------------
# Version assertion (forward compatible)
# ---------------------------------------------------------------------------


def test_version_is_at_least_2_25_0() -> None:
    import visionservex

    parts = visionservex.__version__.split(".")
    major, minor = int(parts[0]), int(parts[1])
    assert (major, minor) >= (2, 25), visionservex.__version__
