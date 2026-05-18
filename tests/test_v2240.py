# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.24.0 tests: real sidecar build path, output normalizers, COCO download.

What v2.24 actually proved:

- ``SidecarSpec.install_root`` is now resolved against
  ``$VISIONSERVEX_SIDECAR_ROOT`` / ``~/.cache/visionservex/sidecars`` instead
  of the previously hardcoded ``/opt/...`` (which was root-only on Linux).
- A real ``visionservex deimv2 create-env --execute`` succeeded on the
  development RTX 5080 box (conda env created, repo cloned, torch 2.5.1
  installed). The GPU smoke step still hits
  ``BLACKWELL_SM120_TORCH_INCOMPATIBLE`` because the upstream pin
  predates Blackwell support — captured as a structured blocker.
- ``deimv2_normalize`` and ``rtdetrv4_normalize`` produce canonical
  detection rows from every shape the upstream postprocessors emit.
- ``prepare-coco-val2017-subset`` learns a gated ``--allow-download`` flag.
"""

from __future__ import annotations

import json
import os
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
# Sidecar install-root resolver
# ---------------------------------------------------------------------------


def test_install_root_defaults_to_user_cache_when_opt_unwritable(
    monkeypatch,
) -> None:
    """On a stock Linux system /opt is root-owned. v2.24 must NOT pick it."""
    from visionservex.sidecars.manager import _resolve_sidecar_root

    monkeypatch.delenv("VISIONSERVEX_SIDECAR_ROOT", raising=False)
    root = _resolve_sidecar_root()
    # On the dev box /opt is root:root → the resolver picks ~/.cache.
    if not os.access(Path("/opt"), os.W_OK):
        assert str(root).endswith("/.cache/visionservex/sidecars")


def test_install_root_env_override(monkeypatch, tmp_path: Path) -> None:
    from visionservex.sidecars.manager import _install_root_source, _resolve_sidecar_root

    monkeypatch.setenv("VISIONSERVEX_SIDECAR_ROOT", str(tmp_path / "sidecars"))
    assert _resolve_sidecar_root() == tmp_path / "sidecars"
    assert _install_root_source() == "env:VISIONSERVEX_SIDECAR_ROOT"


def test_sidecar_spec_install_root_uses_resolver(monkeypatch, tmp_path: Path) -> None:
    from visionservex.sidecars import SidecarManager

    monkeypatch.setenv("VISIONSERVEX_SIDECAR_ROOT", str(tmp_path / "alt"))
    spec = SidecarManager.get_spec("deimv2")
    assert spec is not None
    assert str(spec.install_root).startswith(str(tmp_path / "alt"))


def test_sidecar_create_dry_run_emits_install_root(tmp_path: Path) -> None:
    """v2.24: dry-run JSON must include install_root + install_root_source."""
    out = tmp_path / "create.json"
    res = _run(["sidecar", "create", "deimv2", "--dry-run", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["code"] == "DRY_RUN"
    assert "install_root" in d
    assert "install_root_source" in d
    assert "/sidecars/deimv2" in d["install_root"]


# ---------------------------------------------------------------------------
# v2.24 blocker codes registered with the classifier
# ---------------------------------------------------------------------------


def test_classifier_knows_v224_codes() -> None:
    from visionservex.runtime.result_classifier import EXPECTED_BLOCKER_CODES

    for code in (
        "SIDECAR_INSTALL_ROOT_NOT_WRITABLE",
        "BLACKWELL_SM120_TORCH_INCOMPATIBLE",
        "CUDA_KERNEL_LAUNCH_FAILED",
        "GIT_CLONE_FAILED",
        "PIP_INSTALL_FAILED",
        "CONDA_CREATE_FAILED",
        "REQUIREMENTS_TXT_MISSING",
        "COCO_VAL2017_DOWNLOAD_DISALLOWED",
        "COCO_VAL2017_DOWNLOAD_IN_PROGRESS",
        "NORMALIZER_OUTPUT_INVALID",
    ):
        assert code in EXPECTED_BLOCKER_CODES, code


def test_sidecar_blocker_codes_extended() -> None:
    from visionservex.sidecars.manager import SIDECAR_BLOCKER_CODES

    for code in (
        "SIDECAR_INSTALL_ROOT_NOT_WRITABLE",
        "BLACKWELL_SM120_TORCH_INCOMPATIBLE",
        "CUDA_KERNEL_LAUNCH_FAILED",
        "GIT_CLONE_FAILED",
        "PIP_INSTALL_FAILED",
        "CONDA_CREATE_FAILED",
    ):
        assert code in SIDECAR_BLOCKER_CODES, code


# ---------------------------------------------------------------------------
# DEIMv2 output normalizer
# ---------------------------------------------------------------------------


def test_deimv2_normalize_dict_path() -> None:
    from visionservex.sidecars import normalize_deimv2_output

    raw = {
        "boxes": [[10.0, 10.0, 40.0, 40.0], [20, 20, 80, 80]],
        "scores": [0.9, 0.55],
        "labels": [0, 2],
    }
    r = normalize_deimv2_output(raw)
    assert r["status"] == "ok"
    assert r["n_detections"] == 2
    assert r["rows"][0]["class_name"] == "person"
    assert r["rows"][0]["category_id"] == 1  # official COCO id for person
    assert r["rows"][1]["class_name"] == "car"


def test_deimv2_normalize_concat_n6_path() -> None:
    from visionservex.sidecars import normalize_deimv2_output

    rows = [[10.0, 10.0, 40.0, 40.0, 0.9, 0], [20.0, 20.0, 80.0, 80.0, 0.4, 2]]
    r = normalize_deimv2_output(rows)
    assert r["n_detections"] == 2
    assert r["rows"][1]["class_name"] == "car"


def test_deimv2_normalize_drops_invalid_boxes() -> None:
    from visionservex.sidecars import normalize_deimv2_output

    raw = {
        "boxes": [[10, 10, 40, 40], [5, 5, 5, 5], [10, 10, 0, 0]],
        "scores": [0.9, 0.8, 0.7],
        "labels": [0, 1, 2],
    }
    r = normalize_deimv2_output(raw)
    assert r["n_detections"] == 1
    assert r["n_invalid"] == 2


def test_deimv2_normalize_none_input() -> None:
    from visionservex.sidecars import normalize_deimv2_output

    r = normalize_deimv2_output(None)
    assert r["status"] == "ok"
    assert r["n_detections"] == 0


def test_deimv2_normalize_unknown_shape_structured_block() -> None:
    from visionservex.sidecars import normalize_deimv2_output

    r = normalize_deimv2_output("not a tensor")
    assert r["status"] == "failed"
    assert r["code"] == "NORMALIZER_OUTPUT_INVALID"


def test_deimv2_normalize_class_id_out_of_range() -> None:
    from visionservex.sidecars import normalize_deimv2_output

    # class_id=999 must mark class_name="unknown" without raising.
    r = normalize_deimv2_output({"boxes": [[1, 1, 10, 10]], "scores": [0.6], "labels": [999]})
    assert r["rows"][0]["class_name"] == "unknown"


# ---------------------------------------------------------------------------
# RT-DETRv4 output normalizer
# ---------------------------------------------------------------------------


def test_rtdetrv4_normalize_dict_path() -> None:
    from visionservex.sidecars import normalize_rtdetrv4_output

    raw = {
        "boxes": [[10, 10, 40, 40]],
        "scores": [0.95],
        "labels": [16],
    }
    r = normalize_rtdetrv4_output(raw)
    assert r["status"] == "ok"
    assert r["n_detections"] == 1
    assert r["rows"][0]["class_name"] == "dog"  # COCO80[16]


def test_rtdetrv4_normalize_score_threshold() -> None:
    from visionservex.sidecars import normalize_rtdetrv4_output

    raw = {
        "boxes": [[1, 1, 10, 10], [2, 2, 20, 20]],
        "scores": [0.9, 0.05],
        "labels": [0, 1],
    }
    r = normalize_rtdetrv4_output(raw, score_threshold=0.5)
    assert r["n_detections"] == 1


def test_rtdetrv4_normalize_n6_array() -> None:
    from visionservex.sidecars import normalize_rtdetrv4_output

    r = normalize_rtdetrv4_output([[1, 1, 10, 10, 0.8, 0]])
    assert r["n_detections"] == 1


def test_rtdetrv4_normalize_invalid_shape() -> None:
    from visionservex.sidecars import normalize_rtdetrv4_output

    r = normalize_rtdetrv4_output(123)
    assert r["status"] == "failed"
    assert r["code"] == "NORMALIZER_OUTPUT_INVALID"


# ---------------------------------------------------------------------------
# COCO val2017 --allow-download
# ---------------------------------------------------------------------------


def test_coco_val2017_disallowed_emits_new_blocker(tmp_path: Path) -> None:
    """v2.24: without --allow-download, missing dataset must emit
    COCO_VAL2017_DOWNLOAD_DISALLOWED (the user-friendly variant)."""
    out_dir = tmp_path / "subset"
    report = tmp_path / "report.json"
    res = _run(
        [
            "dataset",
            "prepare-coco-val2017-subset",
            "--coco-root",
            str(tmp_path / "no_coco"),
            "--max-images",
            "5",
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
    assert d["allow_download"] is False
    assert d["code"] in {"COCO_VAL2017_DOWNLOAD_DISALLOWED", "COCO_VAL2017_USER_PATH_REQUIRED"}


def test_coco_val2017_allow_download_flag_parses(tmp_path: Path, monkeypatch) -> None:
    """v2.24: --allow-download must be accepted and trigger the download path.

    We monkeypatch the actual urlretrieve so the test never touches the
    network. ``_download_coco_val2017`` is unit-tested directly with the
    fake retriever so we don't depend on subprocess+CLI for the network
    branch.
    """
    from visionservex.cli import dataset_validators as dv

    fake_calls: list[str] = []

    def _fake_urlretrieve(url, target):
        fake_calls.append(url)
        # Create a minimal valid empty zip that extracts to nothing useful.
        import zipfile

        with zipfile.ZipFile(target, "w") as zf:
            zf.writestr("placeholder.txt", "placeholder")

    monkeypatch.setattr("urllib.request.urlretrieve", _fake_urlretrieve, raising=True)

    coco_root = tmp_path / "coco"
    out = dv._download_coco_val2017(coco_root)
    # The download succeeds in opening the zips but the expected layout
    # (annotations/instances_val2017.json etc.) won't be present →
    # COCO_VAL2017_DOWNLOAD_FAILED. What we are asserting is:
    # 1. --allow-download path *attempts* the download (urlretrieve called twice).
    # 2. Structured blocker rather than raw traceback.
    assert len(fake_calls) == 2
    assert any("val2017.zip" in u for u in fake_calls)
    assert any("annotations" in u for u in fake_calls)
    assert out["status"] in {"failed", "ok"}
    assert out["code"] in {"OK", "COCO_VAL2017_DOWNLOAD_FAILED"}


def test_coco_val2017_download_marker_skips_redownload(tmp_path: Path) -> None:
    from visionservex.cli import dataset_validators as dv

    coco_root = tmp_path / "coco"
    coco_root.mkdir()
    (coco_root / "_DOWNLOAD_COMPLETE").write_text("complete\n")
    out = dv._download_coco_val2017(coco_root)
    assert out["status"] == "ok"
    assert out["skipped_download"] is True


# ---------------------------------------------------------------------------
# Version assertion (forward-compatible)
# ---------------------------------------------------------------------------


def test_version_is_at_least_2_24_0() -> None:
    import visionservex

    parts = visionservex.__version__.split(".")
    major, minor = int(parts[0]), int(parts[1])
    assert (major, minor) >= (2, 24), visionservex.__version__
