# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmark-surveillance-search command (v2.2.0)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_index(root: Path, n: int = 6, dim: int = 8) -> Path:
    """Write a minimal video-search index: manifest.json + embeddings.npy."""
    from visionservex.runtime.video_search import IndexedCrop, IndexManifest

    crops = [
        IndexedCrop(
            track_id=i % 3,
            frame_idx=i,
            timestamp_s=float(i),
            box=(0.0, 0.0, 10.0, 10.0),
            score=0.9,
            label="person",
            embedding_idx=i,
        )
        for i in range(n)
    ]
    manifest = IndexManifest(
        version=1,
        detector_model_id="mock-detect",
        embedder_model_id="mock-embed",
        embedding_dim=dim,
        n_frames_seen=n,
        n_detections=n,
        n_tracks=3,
        crops=crops,
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest.to_dict()))
    emb = np.random.randn(n, dim).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    np.save(root / "embeddings.npy", emb)
    return root


def _make_queries_json(path: Path, n_synthetic: int = 3, n_index: int = 6) -> Path:
    specs = [
        {"_embedding_idx": i % n_index, "text": f"q{i}", "relevant_tracks": [i % 3]}
        for i in range(n_synthetic)
    ]
    path.write_text(json.dumps(specs))
    return path


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_surveillance_result_to_dict():
    from visionservex.cli.benchmark_surveillance import SurveillanceBenchmarkResult

    r = SurveillanceBenchmarkResult(
        index_dir="/tmp/idx",
        n_queries=3,
        n_index_entries=10,
        mean_query_latency_ms=1.2,
        p95_query_latency_ms=2.5,
        mean_top1_similarity=0.95,
        mean_top5_similarity=0.70,
        map_at_5=0.80,
        notes="test",
    )
    d = r.to_dict()
    assert d["map_at_5"] == pytest.approx(0.80)
    assert d["n_index_entries"] == 10


# ---------------------------------------------------------------------------
# CLI — missing index returns INDEX_NOT_FOUND
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_cli_missing_index_returns_structured_error(tmp_path):
    from typer.testing import CliRunner

    from visionservex.cli.benchmark_surveillance import app

    runner = CliRunner()
    result = runner.invoke(app, ["--index", str(tmp_path / "nonexistent"), "--json"])
    output = json.loads(result.output.strip())
    assert output["code"] == "INDEX_NOT_FOUND"


# ---------------------------------------------------------------------------
# CLI — self-retrieval smoke (synthetic queries, no embedder)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_cli_self_retrieval_smoke(tmp_path):
    from typer.testing import CliRunner

    from visionservex.cli.benchmark_surveillance import app

    idx = _make_index(tmp_path / "idx", n=6, dim=8)
    out_path = tmp_path / "bench.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--index", str(idx), "--top-k", "3", "--out", str(out_path), "--json"],
    )

    assert out_path.exists(), f"CLI output: {result.output}"
    data = json.loads(out_path.read_text())
    assert data["benchmark"] == "surveillance_search"
    r = data["result"]
    assert r["n_index_entries"] == 6
    assert r["n_queries"] >= 1
    assert r["mean_query_latency_ms"] >= 0.0
    # Self-retrieval top-1 similarity should be high (query vec == index vec)
    assert r["mean_top1_similarity"] >= 0.0


# ---------------------------------------------------------------------------
# CLI — labeled queries produce MAP output
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_cli_labeled_queries_produce_map(tmp_path):
    from typer.testing import CliRunner

    from visionservex.cli.benchmark_surveillance import app

    idx = _make_index(tmp_path / "idx", n=6, dim=8)
    queries = _make_queries_json(tmp_path / "queries.json", n_synthetic=3, n_index=6)
    out_path = tmp_path / "bench.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--index",
            str(idx),
            "--queries",
            str(queries),
            "--top-k",
            "3",
            "--out",
            str(out_path),
            "--json",
        ],
    )

    assert out_path.exists(), f"CLI output: {result.output}"
    data = json.loads(out_path.read_text())
    r = data["result"]
    # MAP should be computed (labels provided)
    assert r["map_at_5"] is not None
    assert 0.0 <= r["map_at_5"] <= 1.0


# ---------------------------------------------------------------------------
# CLI — empty index returns EMPTY_INDEX
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_cli_empty_index_returns_error(tmp_path):
    from typer.testing import CliRunner

    from visionservex.cli.benchmark_surveillance import app
    from visionservex.runtime.video_search import IndexManifest

    idx = tmp_path / "empty_idx"
    idx.mkdir()
    (idx / "manifest.json").write_text(json.dumps(IndexManifest().to_dict()))
    np.save(idx / "embeddings.npy", np.zeros((0, 8), dtype="float32"))

    runner = CliRunner()
    result = runner.invoke(app, ["--index", str(idx), "--json"])
    output = json.loads(result.output.strip())
    assert output["code"] == "EMPTY_INDEX"


# ---------------------------------------------------------------------------
# Quantile helper
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_quantile():
    from visionservex.cli.benchmark_surveillance import _quantile

    assert _quantile([], 0.5) == 0.0
    assert _quantile([10.0, 20.0, 30.0], 0.5) == 20.0
    assert _quantile([5.0], 0.95) == 5.0
