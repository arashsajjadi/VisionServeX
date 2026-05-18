# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.23.0 tests: sidecar manager + DEIMv2/RT-DETRv4 sidecar CLIs + COCO val2017 subset + synthetic gens."""

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
# Sidecar manager + CLI
# ---------------------------------------------------------------------------


def test_sidecar_list_contains_deimv2_and_rtdetrv4(tmp_path: Path) -> None:
    out = tmp_path / "sidecars.json"
    res = _run(["sidecar", "list", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["n_specs"] >= 2
    assert "deimv2" in d["specs"]
    assert "rtdetrv4" in d["specs"]
    for name in ("deimv2", "rtdetrv4"):
        spec = d["specs"][name]
        assert spec["torch_version"]
        assert spec["upstream_repo"]
        assert spec["license"] == "Apache-2.0"


def test_sidecar_doctor_deimv2_when_env_missing(tmp_path: Path) -> None:
    out = tmp_path / "doctor.json"
    res = _run(["sidecar", "doctor", "deimv2", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["sidecar"] == "deimv2"
    # Either env_exists=True (highly unlikely in test env) or expected_blocker.
    assert d["status"] in {"ok", "expected_blocker"}
    if d["status"] == "expected_blocker":
        assert d["code"] in {"SIDECAR_ENV_MISSING", "CONDA_NOT_AVAILABLE"}


def test_sidecar_create_dry_run_emits_planned_commands(tmp_path: Path) -> None:
    out = tmp_path / "create.json"
    res = _run(["sidecar", "create", "deimv2", "--dry-run", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["code"] == "DRY_RUN"
    cmds = d.get("planned_commands", [])
    assert any("conda create -n visionservex-deimv2-sidecar" in c for c in cmds)
    assert any("git clone" in c for c in cmds)


def test_sidecar_create_unknown_spec(tmp_path: Path) -> None:
    out = tmp_path / "create.json"
    res = _run(
        ["sidecar", "create", "bogus-sidecar", "--dry-run", "--format", "json", "--out", str(out)]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["status"] == "failed"
    assert d["code"] == "SIDECAR_ENV_MISSING"


# ---------------------------------------------------------------------------
# RT-DETRv4 — obsolete blocker replaced
# ---------------------------------------------------------------------------


def test_rtdetrv4_doctor_no_longer_returns_obsolete_blocker(tmp_path: Path) -> None:
    out = tmp_path / "rtv4_doctor.json"
    res = _run(["rtdetrv4", "doctor", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    d = json.loads(out.read_text())
    # The v2.22 obsolete blocker must NOT be the current code.
    assert d["code"] != "RTDETRV4_UPSTREAM_NOT_RELEASED"
    assert d["v2_22_obsolete_blocker_replaced"] == "RTDETRV4_UPSTREAM_NOT_RELEASED"
    assert d["upstream_repo"] == "https://github.com/RT-DETRs/RT-DETRv4"
    assert d["license"] == "Apache-2.0"


def test_rtdetrv4_pull_emits_gdown_command(tmp_path: Path) -> None:
    out = tmp_path / "rtv4_pull.json"
    res = _run(["rtdetrv4", "pull", "rtdetrv4-x", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["code"] == "CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP"
    assert "gdown" in d["gdown_command"]
    assert d["reported_AP"] == 57.0


def test_rtdetrv4_smoke_test_returns_sidecar_env_missing_when_uninstalled(tmp_path: Path) -> None:
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
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["status"] == "expected_blocker"
    # In a fresh test env the sidecar env is missing.
    assert d["code"] in {"SIDECAR_ENV_MISSING", "CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP"}


def test_rtdetrv4_tensorrt_backend_blocked(tmp_path: Path) -> None:
    """The TensorRT path on RTX 5080 has an open accuracy bug; smoke-test must structured-block."""
    from PIL import Image as _PIL

    img = tmp_path / "img.jpg"
    _PIL.new("RGB", (32, 32)).save(img)
    out = tmp_path / "smoke.json"
    # If the sidecar isn't installed, we get SIDECAR_ENV_MISSING first.
    # When the env IS present, the TensorRT branch returns
    # RTDETRV4_TRT_5080_ACCURACY_BUG_OPEN.
    res = _run(
        [
            "rtdetrv4",
            "smoke-test",
            "rtdetrv4-s",
            str(img),
            "--backend",
            "tensorrt",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["status"] == "expected_blocker"
    assert d["code"] in {
        "RTDETRV4_TRT_5080_ACCURACY_BUG_OPEN",
        "SIDECAR_ENV_MISSING",
        "CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP",
    }


# ---------------------------------------------------------------------------
# DEIMv2 — create-env subcommand
# ---------------------------------------------------------------------------


def test_deimv2_create_env_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "create.json"
    res = _run(["deimv2", "create-env", "--dry-run", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["code"] == "DRY_RUN"
    cmds = d.get("planned_commands", [])
    assert any("torch==2.5.1" in c for c in cmds)
    assert any("Intellindust-AI-Lab/DEIMv2" in c for c in cmds)


# ---------------------------------------------------------------------------
# COCO val2017 400-subset CLI
# ---------------------------------------------------------------------------


def test_coco_val2017_subset_missing_path_returns_structured_blocker(tmp_path: Path) -> None:
    """v2.23 contract preserved through v2.24: missing dataset without
    --allow-download must surface a structured blocker. v2.24 splits the
    code into ``COCO_VAL2017_DOWNLOAD_DISALLOWED`` (when --coco-root does not
    exist at all) vs ``COCO_VAL2017_USER_PATH_REQUIRED`` (when the dir exists
    but the layout is wrong)."""
    out_dir = tmp_path / "subset"
    report = tmp_path / "report.json"
    res = _run(
        [
            "dataset",
            "prepare-coco-val2017-subset",
            "--coco-root",
            str(tmp_path / "no_coco"),
            "--max-images",
            "400",
            "--out",
            str(out_dir),
            "--report",
            str(report),
            "--format",
            "json",
        ]
    )
    assert res.returncode == 2
    d = json.loads(report.read_text())
    assert d["code"] in {
        "COCO_VAL2017_USER_PATH_REQUIRED",
        "COCO_VAL2017_DOWNLOAD_DISALLOWED",
    }


# ---------------------------------------------------------------------------
# Synthetic dataset generators
# ---------------------------------------------------------------------------


def test_generate_synthetic_anomaly_defect(tmp_path: Path) -> None:
    out_dir = tmp_path / "anomaly"
    res = _run(
        [
            "dataset",
            "generate-synthetic",
            "anomaly-defect",
            "--out",
            str(out_dir),
            "--n-samples",
            "3",
            "--format",
            "json",
        ]
    )
    assert res.returncode == 0
    assert (out_dir / "normal").exists()
    assert (out_dir / "test").exists()
    assert (out_dir / "_SYNTHETIC_MANIFEST.json").exists()
    manifest = json.loads((out_dir / "_SYNTHETIC_MANIFEST.json").read_text())
    assert manifest["kind"] == "anomaly-defect"
    assert manifest["n_normal"] == 3


def test_generate_synthetic_agriculture_hbb(tmp_path: Path) -> None:
    out_dir = tmp_path / "agri"
    res = _run(
        [
            "dataset",
            "generate-synthetic",
            "agriculture-hbb",
            "--out",
            str(out_dir),
            "--n-samples",
            "2",
            "--format",
            "json",
        ]
    )
    assert res.returncode == 0
    assert (out_dir / "images").exists()
    assert (out_dir / "labels").exists()
    assert (out_dir / "data.yaml").exists()


def test_generate_synthetic_aerial_obb(tmp_path: Path) -> None:
    out_dir = tmp_path / "aerial"
    res = _run(
        [
            "dataset",
            "generate-synthetic",
            "aerial-obb",
            "--out",
            str(out_dir),
            "--n-samples",
            "2",
            "--format",
            "json",
        ]
    )
    assert res.returncode == 0
    assert (out_dir / "labelTxt").exists()


def test_generate_synthetic_medical_nifti(tmp_path: Path) -> None:
    out_dir = tmp_path / "med"
    res = _run(
        [
            "dataset",
            "generate-synthetic",
            "medical-nifti",
            "--out",
            str(out_dir),
            "--n-samples",
            "2",
            "--format",
            "json",
        ]
    )
    assert res.returncode == 0
    assert (out_dir / "boxes.json").exists()


def test_generate_synthetic_unknown_kind(tmp_path: Path) -> None:
    out_dir = tmp_path / "x"
    _run(
        [
            "dataset",
            "generate-synthetic",
            "unknown-thing",
            "--out",
            str(out_dir),
            "--n-samples",
            "1",
            "--format",
            "json",
        ]
    )
    # The CLI succeeds at file-system level but emits structured failure inside the manifest.
    manifest = json.loads((out_dir / "_SYNTHETIC_MANIFEST.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["code"] == "UNKNOWN_SYNTHETIC_KIND"


# ---------------------------------------------------------------------------
# result_classifier v2.23 codes
# ---------------------------------------------------------------------------


def test_classifier_knows_v223_codes() -> None:
    from visionservex.runtime.result_classifier import EXPECTED_BLOCKER_CODES

    for code in (
        "SIDECAR_ENV_MISSING",
        "SIDECAR_CREATE_FAILED",
        "SIDECAR_TIMEOUT",
        "CUSTOM_OPS_COMPILATION",
        "CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP",
        "CONDA_NOT_AVAILABLE",
        "MVTEC_NONCOMMERCIAL",
        "DOTA_NONCOMMERCIAL",
        "RTDETRV4_TRT_5080_ACCURACY_BUG_OPEN",
        "COCO_VAL2017_USER_PATH_REQUIRED",
    ):
        assert code in EXPECTED_BLOCKER_CODES, code
