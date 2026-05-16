# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmark-anomaly command (v2.2.0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mvtec_fixture(root: Path, n_normal: int = 4, n_anomaly: int = 2) -> Path:
    """Create a tiny MVTec-like directory layout."""
    for sub in ("train/good", "test/good", "test/crack"):
        (root / sub).mkdir(parents=True)
    for i in range(n_normal):
        img = Image.new("RGB", (32, 32), color=(200, 200, 200))
        img.save(root / "train" / "good" / f"n_{i:02d}.png")
        img.save(root / "test" / "good" / f"tn_{i:02d}.png")
    for i in range(n_anomaly):
        img = Image.new("RGB", (32, 32), color=(50, 0, 0))
        img.save(root / "test" / "crack" / f"a_{i:02d}.png")
    return root


def _make_simple_fixture(root: Path, n_normal: int = 4, n_test: int = 2) -> Path:
    """Create a simple normal/test layout."""
    (root / "normal").mkdir(parents=True)
    (root / "test").mkdir(parents=True)
    for i in range(n_normal):
        img = Image.new("RGB", (32, 32), color=(180, 180, 180))
        img.save(root / "normal" / f"n_{i:02d}.png")
    for i in range(n_test):
        img = Image.new("RGB", (32, 32), color=(80, 0, 0))
        img.save(root / "test" / f"t_{i:02d}.png")
    return root


# ---------------------------------------------------------------------------
# AUROC helper
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_compute_auroc_perfect_separation():
    from visionservex.cli.benchmark_anomaly_cmd import _compute_auroc

    normal = [0.1, 0.2, 0.15]
    anomaly = [0.8, 0.9, 0.85]
    auroc = _compute_auroc(normal, anomaly)
    assert auroc is not None
    assert auroc == pytest.approx(1.0, abs=0.01)


@pytest.mark.fast
def test_compute_auroc_empty_returns_none():
    from visionservex.cli.benchmark_anomaly_cmd import _compute_auroc

    assert _compute_auroc([], [0.8]) is None
    assert _compute_auroc([0.1], []) is None


@pytest.mark.fast
def test_compute_auroc_random_chance():
    from visionservex.cli.benchmark_anomaly_cmd import _compute_auroc

    normal = [0.5, 0.5, 0.5]
    anomaly = [0.5, 0.5, 0.5]
    auroc = _compute_auroc(normal, anomaly)
    assert auroc is not None
    assert 0.0 <= auroc <= 1.0


# ---------------------------------------------------------------------------
# CLI — missing anomalib returns ANOMALIB_REQUIRED
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_cli_missing_anomalib_returns_structured_error(tmp_path):
    from unittest.mock import patch

    from typer.testing import CliRunner

    from visionservex.cli.benchmark_anomaly_cmd import app

    ds_root = _make_simple_fixture(tmp_path / "ds")
    runner = CliRunner()
    with patch("importlib.import_module", side_effect=ImportError("anomalib not installed")):
        result = runner.invoke(app, ["--dataset", f"simple:{ds_root}", "--json"])

    output = json.loads(result.output.strip())
    assert output["code"] == "ANOMALIB_REQUIRED"
    assert "fix" in output
    assert "visionservex[anomaly]" in output["fix"]


# ---------------------------------------------------------------------------
# CLI — missing dataset path returns ANOMALY_DATASET_REQUIRED
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_cli_missing_dataset_path(tmp_path):
    import importlib

    from typer.testing import CliRunner

    from visionservex.cli.benchmark_anomaly_cmd import app

    # Patch anomalib to appear available, but dataset path does not exist
    mock_module = object()
    real_import = importlib.import_module

    def _patched_import(name, *args, **kwargs):
        if name == "anomalib":
            return mock_module
        return real_import(name, *args, **kwargs)

    from unittest.mock import patch

    runner = CliRunner()
    with patch("importlib.import_module", side_effect=_patched_import):
        result = runner.invoke(app, ["--dataset", f"mvtec:{tmp_path / 'nonexistent'}", "--json"])

    output = json.loads(result.output.strip())
    assert output["code"] == "ANOMALY_DATASET_REQUIRED"


# ---------------------------------------------------------------------------
# CLI — MVTec fixture exists, anomalib fails gracefully
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_cli_mvtec_fixture_anomalib_api_failure(tmp_path):
    """When anomalib Engine raises, result.error is set — not a traceback crash."""
    import importlib

    from typer.testing import CliRunner

    from visionservex.cli.benchmark_anomaly_cmd import app

    ds_root = _make_mvtec_fixture(tmp_path / "ds")
    out_path = tmp_path / "bench.json"

    real_import = importlib.import_module

    def _patched_import(name, *args, **kwargs):
        if name == "anomalib":
            import types

            m = types.ModuleType("anomalib")
            return m
        if name == "anomalib.data":
            raise ImportError("anomalib.data not available in mock")
        return real_import(name, *args, **kwargs)

    from unittest.mock import patch

    runner = CliRunner()
    with patch("importlib.import_module", side_effect=_patched_import):
        result = runner.invoke(
            app,
            ["--dataset", f"mvtec:{ds_root}", "--json", "--out", str(out_path)],
        )

    # The CLI must not crash; it must produce valid JSON
    output = json.loads(result.output.strip())
    assert "result" in output
    r = output["result"]
    assert "error" in r
    assert out_path.exists()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_anomaly_result_to_dict():
    from visionservex.cli.benchmark_anomaly_cmd import AnomalyBenchmarkResult

    r = AnomalyBenchmarkResult(
        model="patchcore",
        dataset="mvtec:/tmp/ds",
        n_normal_train=10,
        n_normal_test=5,
        n_anomaly_test=3,
        image_auroc=0.87,
        anomaly_score_mean_normal=0.2,
        anomaly_score_mean_anomaly=0.8,
        score_separation=0.6,
        latency_p50_ms=12.0,
        latency_p95_ms=18.0,
        heatmaps_saved=0,
        notes="test",
    )
    d = r.to_dict()
    assert d["image_auroc"] == pytest.approx(0.87)
    assert d["score_separation"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# iter_images helper
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_iter_images(tmp_path):
    from visionservex.cli.benchmark_anomaly_cmd import _iter_images

    for i in range(3):
        Image.new("RGB", (8, 8)).save(tmp_path / f"img_{i}.png")
    (tmp_path / "file.txt").write_text("not an image")

    result = _iter_images(tmp_path)
    assert len(result) == 3
    assert all(p.suffix == ".png" for p in result)
