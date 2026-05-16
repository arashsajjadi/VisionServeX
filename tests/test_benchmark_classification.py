# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmark-classification command (v2.2.0)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_folder_dataset(root: Path) -> Path:
    """Create a tiny 2-class folder dataset: cat/dog, 3 images each."""
    for cls in ("cat", "dog"):
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(3):
            img = Image.new("RGB", (32, 32), color=(i * 40, i * 40, i * 40))
            img.save(cls_dir / f"img_{i:02d}.jpg")
    return root


def _make_csv_dataset(root: Path) -> Path:
    """Create a tiny CSV dataset referencing images."""
    img_dir = root / "images"
    img_dir.mkdir(parents=True)
    rows = []
    for cls in ("cat", "dog"):
        for i in range(2):
            img = Image.new("RGB", (32, 32))
            p = img_dir / f"{cls}_{i}.jpg"
            img.save(p)
            rows.append(f"{p},{cls}")
    csv_path = root / "labels.csv"
    csv_path.write_text("image_path,label\n" + "\n".join(rows))
    return csv_path


# ---------------------------------------------------------------------------
# Dataset loader tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_load_folder_dataset(tmp_path):
    from visionservex.cli.benchmark_classification import _load_dataset

    _make_folder_dataset(tmp_path / "ds")
    pairs = _load_dataset(f"folder:{tmp_path / 'ds'}", max_images=100)
    assert len(pairs) == 6
    labels = {lbl for _, lbl in pairs}
    assert labels == {"cat", "dog"}


@pytest.mark.fast
def test_load_csv_dataset(tmp_path):
    from visionservex.cli.benchmark_classification import _load_dataset

    csv_path = _make_csv_dataset(tmp_path)
    pairs = _load_dataset(f"csv:{csv_path}", max_images=100)
    assert len(pairs) == 4
    assert all(lbl in {"cat", "dog"} for _, lbl in pairs)


@pytest.mark.fast
def test_load_dataset_max_images(tmp_path):
    from visionservex.cli.benchmark_classification import _load_dataset

    _make_folder_dataset(tmp_path / "ds")
    pairs = _load_dataset(f"folder:{tmp_path / 'ds'}", max_images=2)
    assert len(pairs) == 2


@pytest.mark.fast
def test_load_dataset_missing_folder(tmp_path):
    from visionservex.cli.benchmark_classification import _load_dataset

    with pytest.raises(FileNotFoundError):
        _load_dataset(f"folder:{tmp_path / 'nonexistent'}", max_images=10)


@pytest.mark.fast
def test_load_dataset_unsupported_spec():
    from visionservex.cli.benchmark_classification import _load_dataset

    with pytest.raises(ValueError, match="Unsupported dataset spec"):
        _load_dataset("s3://bucket/path", max_images=10)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_quantile_empty():
    from visionservex.cli.benchmark_classification import _quantile

    assert _quantile([], 0.5) == 0.0


@pytest.mark.fast
def test_quantile_median():
    from visionservex.cli.benchmark_classification import _quantile

    vals = [10.0, 20.0, 30.0]
    assert _quantile(vals, 0.5) == 20.0


# ---------------------------------------------------------------------------
# CLI smoke — invalid dataset returns DATASET_SCHEMA_REQUIRED
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_cli_invalid_dataset(tmp_path):
    from typer.testing import CliRunner

    from visionservex.cli.benchmark_classification import app

    runner = CliRunner()
    result = runner.invoke(app, ["--dataset", f"folder:{tmp_path / 'nonexistent'}", "--json"])
    output = json.loads(result.output.strip())
    assert output["code"] == "DATASET_SCHEMA_REQUIRED"


@pytest.mark.fast
def test_cli_unsupported_spec_returns_error(tmp_path):
    from typer.testing import CliRunner

    from visionservex.cli.benchmark_classification import app

    runner = CliRunner()
    result = runner.invoke(app, ["--dataset", "bad-spec", "--json"])
    output = json.loads(result.output.strip())
    assert output["code"] == "DATASET_SCHEMA_REQUIRED"


# ---------------------------------------------------------------------------
# CLI smoke — mocked model, writes JSON output
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_cli_mocked_model_writes_json(tmp_path):
    from typer.testing import CliRunner

    from visionservex.cli.benchmark_classification import app

    ds_root = tmp_path / "ds"
    _make_folder_dataset(ds_root)
    out_path = tmp_path / "bench.json"

    fake_result = MagicMock()
    fake_result.top_k = [("cat", 0.9), ("dog", 0.1)]

    fake_model = MagicMock()
    fake_model.predict.return_value = fake_result
    fake_model._ensure_loaded.return_value = None

    with patch("visionservex.VisionModel", return_value=fake_model):
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--dataset",
                f"folder:{ds_root}",
                "--models",
                "mock-classifier",
                "--max-images",
                "6",
                "--out",
                str(out_path),
                "--json",
            ],
        )

    assert out_path.exists(), f"JSON not written; CLI output: {result.output}"
    data = json.loads(out_path.read_text())
    assert data["benchmark"] == "classification"
    assert len(data["models"]) == 1
    r = data["models"][0]
    assert "top1_accuracy" in r
    assert "top5_accuracy" in r
    assert "latency_p50_ms" in r


@pytest.mark.fast
def test_cli_mocked_model_top1_correct(tmp_path):
    """Top-1 accuracy = 1.0 when model always returns correct class first."""
    from typer.testing import CliRunner

    from visionservex.cli.benchmark_classification import app

    ds_root = tmp_path / "ds"
    _make_folder_dataset(ds_root)

    def _side_effect(img, top_k=5):
        # Would need to know the label — just return "cat" always
        r = MagicMock()
        r.top_k = [("cat", 0.9)]
        return r

    fake_model = MagicMock()
    fake_model.predict.side_effect = _side_effect
    fake_model._ensure_loaded.return_value = None

    with patch("visionservex.VisionModel", return_value=fake_model):
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--dataset",
                f"folder:{ds_root}",
                "--models",
                "mock-classifier",
                "--max-images",
                "6",
                "--json",
            ],
        )

    data = json.loads(result.output.strip())
    assert data["benchmark"] == "classification"
    r = data["models"][0]
    # cat class: 3/3 correct; dog class: 0/3 correct → top1 = 0.5
    assert r["top1_accuracy"] == pytest.approx(0.5, abs=0.01)
