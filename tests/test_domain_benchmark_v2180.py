# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Phase 2/3/4 (v2.18.0): domain candidates + dataset validators + domain benchmarks."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _vsx() -> list[str]:
    binary = shutil.which("visionservex")
    return [binary] if binary else [sys.executable, "-m", "visionservex"]


def _run(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(_vsx() + args, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------------------
# domain-zoo benchmark-candidates
# ---------------------------------------------------------------------------


def test_domain_candidates_medical(tmp_path: Path) -> None:
    out = tmp_path / "medical.json"
    res = _run(
        [
            "domain-zoo",
            "benchmark-candidates",
            "--domain",
            "medical",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0, res.stderr
    d = json.loads(out.read_text())
    assert d["status"] == "ok"
    assert d["domain"] == "medical"
    assert d["n_rows"] >= 2
    ids = {r["model_id"] for r in d["rows"]}
    assert "medsam" in ids


def test_domain_candidates_agriculture(tmp_path: Path) -> None:
    out = tmp_path / "agri.json"
    res = _run(
        [
            "domain-zoo",
            "benchmark-candidates",
            "--domain",
            "agriculture",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert any(r["benchmark_status"] == "metric_ready" for r in d["rows"])
    # agriclip must be SIDECAR_REQUIRED expected_blocker
    agri = next(r for r in d["rows"] if r["model_id"] == "agriclip")
    assert agri["benchmark_status"] == "expected_blocker"
    assert agri["expected_blocker_code"] == "SIDECAR_REQUIRED"


def test_domain_candidates_unknown_domain(tmp_path: Path) -> None:
    out = tmp_path / "unknown.json"
    res = _run(
        [
            "domain-zoo",
            "benchmark-candidates",
            "--domain",
            "bogus",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 2
    d = json.loads(out.read_text())
    assert d["status"] == "failed"
    assert d["code"] == "UNKNOWN_DOMAIN"


def test_domain_candidates_all_known_domains_return_ok(tmp_path: Path) -> None:
    for domain in (
        "medical",
        "agriculture",
        "aerial",
        "industrial",
        "surveillance",
        "segmentation",
    ):
        out = tmp_path / f"{domain}.json"
        res = _run(
            [
                "domain-zoo",
                "benchmark-candidates",
                "--domain",
                domain,
                "--format",
                "json",
                "--out",
                str(out),
            ]
        )
        assert res.returncode == 0, f"{domain}: {res.stderr}"
        d = json.loads(out.read_text())
        assert d["status"] == "ok"
        assert d["n_rows"] >= 1
        # Each row must carry a dataset_required field
        for r in d["rows"]:
            assert "dataset_required" in r
            assert "metrics_supported" in r
            assert "benchmark_status" in r


# ---------------------------------------------------------------------------
# dataset validators
# ---------------------------------------------------------------------------


def test_validate_anomaly_missing_path(tmp_path: Path) -> None:
    out = tmp_path / "anomaly.json"
    res = _run(
        [
            "dataset",
            "validate-anomaly",
            "--path",
            str(tmp_path / "does_not_exist"),
            "--schema",
            "simple",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 2
    d = json.loads(out.read_text())
    assert d["code"] == "PATH_NOT_FOUND"
    assert d["dataset_type"] == "anomaly"


def test_validate_anomaly_simple_normal_only(tmp_path: Path) -> None:
    root = tmp_path / "anomaly"
    (root / "normal").mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (40, 40), (i * 30, 0, 0)).save(root / "normal" / f"n{i}.png")
    out = tmp_path / "v.json"
    res = _run(
        [
            "dataset",
            "validate-anomaly",
            "--path",
            str(root),
            "--schema",
            "simple",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0  # partial is exit-0
    d = json.loads(out.read_text())
    assert d["status"] == "partial"
    assert d["n_normal_images"] == 3
    assert d["n_defect_images"] == 0


def test_validate_anomaly_simple_with_normal_and_test(tmp_path: Path) -> None:
    root = tmp_path / "anomaly"
    (root / "normal").mkdir(parents=True)
    (root / "test").mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (40, 40), (i * 30, 0, 0)).save(root / "normal" / f"n{i}.png")
    Image.new("RGB", (40, 40), (200, 0, 0)).save(root / "test" / "d0.png")
    out = tmp_path / "v.json"
    res = _run(
        [
            "dataset",
            "validate-anomaly",
            "--path",
            str(root),
            "--schema",
            "simple",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["status"] == "ok"
    assert "image_auroc" in d["metrics_possible"]


def test_validate_agriculture_no_labels(tmp_path: Path) -> None:
    root = tmp_path / "agri"
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (40, 40)).save(images_dir / "a.png")
    out = tmp_path / "v.json"
    res = _run(
        [
            "dataset",
            "validate-agriculture",
            "--path",
            str(root),
            "--task",
            "weed-detection",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["status"] == "partial"
    assert d["blocker_code"] == "LABELS_REQUIRED_FOR_METRICS"
    assert "ap50" not in d["metrics_possible"]


def test_validate_medical_2d_no_box_prompts(tmp_path: Path) -> None:
    root = tmp_path / "med"
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (40, 40)).save(images_dir / "a.png")
    out = tmp_path / "v.json"
    res = _run(
        [
            "dataset",
            "validate-medical",
            "--path",
            str(root),
            "--task",
            "medsam-2d-box",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["status"] == "partial"
    assert d["blocker_code"] == "BOX_PROMPTS_REQUIRED"


def test_validate_medical_3d_nifti_required(tmp_path: Path) -> None:
    root = tmp_path / "med3d"
    root.mkdir()
    out = tmp_path / "v.json"
    res = _run(
        [
            "dataset",
            "validate-medical",
            "--path",
            str(root),
            "--task",
            "totalsegmentator",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["status"] == "expected_blocker"
    assert d["blocker_code"] == "NIFTI_REQUIRED"


def test_validate_aerial_dota_without_obb(tmp_path: Path) -> None:
    root = tmp_path / "aerial"
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (40, 40)).save(root / "images" / "a.png")
    out = tmp_path / "v.json"
    res = _run(
        [
            "dataset",
            "validate-aerial",
            "--path",
            str(root),
            "--dataset-type",
            "dota",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["status"] == "expected_blocker"
    assert d["blocker_code"] == "DOTA_OR_OBB_LABELS_REQUIRED"


def test_validate_surveillance_no_media(tmp_path: Path) -> None:
    root = tmp_path / "surv"
    root.mkdir()
    out = tmp_path / "v.json"
    res = _run(
        [
            "dataset",
            "validate-surveillance",
            "--path",
            str(root),
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 2
    d = json.loads(out.read_text())
    assert d["code"] == "NO_MEDIA_FOUND"


# ---------------------------------------------------------------------------
# domain benchmark commands
# ---------------------------------------------------------------------------


def test_benchmark_medical_returns_not_implemented(tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    res = _run(["benchmark-medical", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["status"] == "expected_blocker"
    assert d["code"] == "BENCHMARK_NOT_IMPLEMENTED"
    assert d["domain"] == "medical"
    # Must NOT promise COCO-AP metrics for medical.
    assert "ap50" not in [m.lower() for m in d.get("metrics_supported", [])]


def test_benchmark_agriculture_needs_labels(tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    res = _run(["benchmark-agriculture", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["code"] == "LABELS_REQUIRED_FOR_METRICS"


def test_benchmark_agriculture_routes_to_detection_with_labels(tmp_path: Path) -> None:
    root = tmp_path / "agri"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    Image.new("RGB", (40, 40)).save(root / "images" / "a.png")
    (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    out = tmp_path / "bench.json"
    res = _run(
        [
            "benchmark-agriculture",
            "--dataset",
            str(root),
            "--models",
            "dfine-s-o365-coco",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["status"] == "ok"
    assert d["code"] == "ROUTED_TO_DETECTION"
    assert "recommended_command" in d


def test_benchmark_aerial_dota_blocked(tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    res = _run(
        ["benchmark-aerial", "--dataset-type", "dota", "--format", "json", "--out", str(out)]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["code"] == "DOTA_OR_OBB_LABELS_REQUIRED"


def test_benchmark_surveillance_needs_source(tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    res = _run(["benchmark-surveillance", "--format", "json", "--out", str(out)])
    assert res.returncode == 2
    d = json.loads(out.read_text())
    assert d["code"] == "NO_MEDIA_FOUND"


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------


def test_concurrency_profile_for_rtx_5080() -> None:
    """On the dev machine, expect desktop_16gb_fast with 2/1/1 workers."""
    from visionservex.runtime.concurrency import build_concurrency_profile

    p = build_concurrency_profile().to_dict()
    # We can't guarantee a specific GPU is present, but the dict shape must be sane.
    assert "gpu_profile" in p
    assert isinstance(p["recommended_small_model_workers"], int)
    assert isinstance(p["recommended_heavy_model_workers"], int)
    assert p["recommended_heavy_model_workers"] >= 1


def test_concurrency_profile_cli(tmp_path: Path) -> None:
    out = tmp_path / "cprof.json"
    res = _run(["dev", "concurrency-profile", "--format", "json", "--out", str(out)])
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert "max_safe_concurrent_requests" in d
    assert "policy" in d
    assert "small_models_parallel" in d["policy"]


def test_benchmark_concurrency_separate_process_returns_not_supported(tmp_path: Path) -> None:
    from visionservex.runtime.concurrency import run_concurrency_benchmark

    result = run_concurrency_benchmark(
        model_id="mock-detect",
        image_paths=[],
        device="cpu",
        require_gpu=False,
        concurrency_levels=[1],
        request_mode="separate-process",
    )
    assert result["status"] == "expected_blocker"
    assert result["code"] == "SHARED_MODEL_CONCURRENCY_NOT_SUPPORTED"


def test_benchmark_concurrency_mock_detect_shared_model(tmp_path: Path) -> None:
    """mock-detect with shared-model concurrency must succeed on CPU."""
    images = []
    for i in range(4):
        p = tmp_path / f"img_{i}.png"
        Image.new("RGB", (40, 40), (i * 50, 0, 0)).save(p)
        images.append(p)
    from visionservex.runtime.concurrency import run_concurrency_benchmark

    result = run_concurrency_benchmark(
        model_id="mock-detect",
        image_paths=images,
        device="cpu",
        require_gpu=False,
        concurrency_levels=[1, 2],
        request_mode="shared-model",
        sample_gpu=False,
    )
    assert result["status"] in ("ok", "partial"), result
    assert len(result["runs"]) == 2
    for r in result["runs"]:
        assert r["n_success"] == 4
        assert "throughput_req_per_sec" in r
        assert "latency_ms_p50" in r
        assert "latency_ms_p95" in r
