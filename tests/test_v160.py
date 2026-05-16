# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tests for v1.6.0: source-grounded manifest, DINOv2 embeddings, domain zoo."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from visionservex.cli.main import app

runner = CliRunner()


# ============================================================
# Source manifest
# ============================================================


def test_manifest_loads():
    from visionservex.model_zoo import SOURCE_MANIFEST, list_all_models

    assert len(SOURCE_MANIFEST) > 20
    assert "dinov2-base" in SOURCE_MANIFEST
    assert "dfine-x-o365-coco" in SOURCE_MANIFEST
    assert "yolo-world" in SOURCE_MANIFEST  # excluded
    ids = list_all_models()
    assert sorted(ids) == ids  # sorted


def test_manifest_dinov2_runnable():
    from visionservex.model_zoo import get_model_source

    src = get_model_source("dinov2-base")
    assert src is not None
    assert src.runnable_in_visionservex is True
    assert src.task == "embed"
    assert src.license == "Apache-2.0"
    assert src.hf_repo == "facebook/dinov2-base"
    assert "github.com/facebookresearch/dinov2" in src.official_repo
    assert src.recommended_action == "add_now"


def test_manifest_yolo_world_excluded():
    from visionservex.model_zoo import get_model_source

    src = get_model_source("yolo-world")
    assert src is not None
    assert src.runnable_in_visionservex is False
    assert src.recommended_action == "do_not_add"
    assert src.license_risk == "restricted"


def test_manifest_deimv2_not_runnable():
    from visionservex.model_zoo import get_model_source

    src = get_model_source("deimv2-s")
    assert src is not None
    assert src.runnable_in_visionservex is False
    assert len(src.known_blockers) > 0
    assert "github.com/Intellindust-AI-Lab/DEIMv2" in src.official_repo


def test_manifest_sam3_external_api():
    from visionservex.model_zoo import get_model_source

    src = get_model_source("sam3-base")
    assert src is not None
    assert src.access_status == "gated"
    assert src.recommended_action == "external_api"


def test_manifest_grounding_dino_15_api_only():
    from visionservex.model_zoo import get_model_source

    src = get_model_source("grounding-dino-1.5-pro")
    assert src is not None
    assert src.access_status == "api_token"
    assert src.license_risk == "api_only"


def test_manifest_no_runnable_without_license():
    from visionservex.model_zoo import SOURCE_MANIFEST

    for mid, src in SOURCE_MANIFEST.items():
        if src.runnable_in_visionservex:
            assert src.license, f"{mid} runnable but no license"


def test_manifest_verify_structure():
    from visionservex.model_zoo import verify_manifest

    report = verify_manifest()
    assert "counts" in report
    assert report["counts"]["total"] > 20
    assert report["counts"]["runnable"] > 5


# ============================================================
# Domain zoo
# ============================================================


def test_domain_zoo_lists_domains():
    from visionservex.model_zoo import list_domains

    domains = list_domains()
    assert "yolo26-competitors" in domains
    assert "sam-family" in domains
    assert "feature-intelligence" in domains
    assert "surveillance" in domains
    assert "medical" in domains
    assert "industrial" in domains


def test_domain_zoo_recommend_feature_intelligence():
    from visionservex.model_zoo import recommend_for_domain

    recipes = recommend_for_domain("feature-intelligence")
    assert len(recipes) >= 1
    assert any("dinov2" in m.lower() for r in recipes for m in r.recommended_models)
    assert any(r.runnable_today for r in recipes)


def test_domain_zoo_recommend_surveillance_not_runnable_today():
    from visionservex.model_zoo import recommend_for_domain

    recipes = recommend_for_domain("surveillance")
    # Surveillance pipeline is roadmap — recipes should exist but runnable_today=False
    assert recipes
    person_search = next((r for r in recipes if "person" in r.goal.lower()), None)
    assert person_search is not None
    assert person_search.runnable_today is False
    assert len(person_search.limitations) > 0


def test_domain_zoo_yolo26_runnable():
    from visionservex.model_zoo import recommend_for_domain

    recipes = recommend_for_domain("yolo26-competitors")
    assert recipes
    assert any(r.runnable_today for r in recipes)
    ids = [m for r in recipes for m in r.recommended_models]
    assert any("dfine-x" in i for i in ids)
    assert any("rfdetr-large" in i for i in ids)


# ============================================================
# CLI: model-zoo
# ============================================================


def test_model_zoo_sources_json():
    result = runner.invoke(app, ["model-zoo", "sources", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) > 20


def test_model_zoo_sources_runnable_only():
    result = runner.invoke(app, ["model-zoo", "sources", "--runnable-only", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    for entry in data:
        assert entry["runnable_in_visionservex"] is True


def test_model_zoo_verify_links_json():
    result = runner.invoke(app, ["model-zoo", "verify-links", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "counts" in data
    assert "issues" in data


def test_model_zoo_show_dinov2():
    result = runner.invoke(app, ["model-zoo", "show", "dinov2-base", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["model_id"] == "dinov2-base"
    assert data["runnable_in_visionservex"] is True


def test_model_zoo_show_not_found():
    result = runner.invoke(app, ["model-zoo", "show", "no-such-model-xyz"])
    assert result.exit_code == 1


def test_model_zoo_export_json(tmp_path):
    out = tmp_path / "manifest.json"
    result = runner.invoke(app, ["model-zoo", "export", "--format", "json", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert "dinov2-base" in data


def test_model_zoo_export_markdown(tmp_path):
    out = tmp_path / "manifest.md"
    result = runner.invoke(app, ["model-zoo", "export", "--format", "markdown", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    text = out.read_text()
    assert "# VisionServeX Model Zoo Manifest" in text


# ============================================================
# CLI: domain-zoo
# ============================================================


def test_domain_zoo_list_json():
    result = runner.invoke(app, ["domain-zoo", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "feature-intelligence" in data
    assert "surveillance" in data


def test_domain_zoo_recommend_yolo26():
    result = runner.invoke(
        app, ["domain-zoo", "recommend", "--domain", "yolo26-competitors", "--json"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) >= 1


def test_domain_zoo_recommend_invalid():
    result = runner.invoke(app, ["domain-zoo", "recommend", "--domain", "no-such-domain"])
    assert result.exit_code != 0


def test_domain_zoo_export_markdown(tmp_path):
    out = tmp_path / "domains.md"
    result = runner.invoke(app, ["domain-zoo", "export", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()


# ============================================================
# Embedding result classes
# ============================================================


def test_embedding_result_to_dict():
    from visionservex.core.embedding_results import EmbeddingResult

    emb = np.random.randn(768).astype(np.float32)
    emb /= np.linalg.norm(emb)
    r = EmbeddingResult(model_id="dinov2-base", task="embed", embedding=emb, embedding_dim=768)
    d = r.to_dict()
    assert d["kind"] == "embedding"
    assert d["embedding_dim"] == 768
    assert "embedding_norm" in d
    assert abs(d["embedding_norm"] - 1.0) < 1e-5


def test_dataset_report_markdown():
    from visionservex.core.embedding_results import DatasetReport

    r = DatasetReport(folder="/tmp/x", model_id="dinov2-base", n_images=10)
    md = r.to_markdown()
    assert "# Dataset Report" in md
    assert "dinov2-base" in md


def test_search_result_to_dict():
    from visionservex.core.embedding_results import SearchHit, SearchResult

    hits = [SearchHit(image_path="a.jpg", score=0.99, rank=1)]
    r = SearchResult(model_id="dinov2-base", query="q.jpg", top_k=10, hits=hits)
    d = r.to_dict()
    assert d["top_k"] == 10
    assert d["hits"][0]["rank"] == 1


def test_similarity_result_to_dict():
    from visionservex.core.embedding_results import SimilarityResult

    r = SimilarityResult(model_id="dinov2-base", image_a="a", image_b="b", cosine_similarity=0.95)
    d = r.to_dict()
    assert d["cosine_similarity"] == 0.95


# ============================================================
# Embedding runtime
# ============================================================


def test_cosine_similarity_unit_vectors():
    from visionservex.runtime.embeddings import cosine_similarity

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_similarity(a, b)) < 1e-5
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-5


def test_cosine_similarity_zero():
    from visionservex.runtime.embeddings import cosine_similarity

    a = np.zeros(3)
    b = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(a, b) == 0.0


def test_embedding_index_save_load(tmp_path):
    from visionservex.runtime.embeddings import EmbeddingIndex

    idx = EmbeddingIndex(
        index_dir=tmp_path / "idx",
        embeddings=np.random.randn(3, 768).astype(np.float32),
        image_paths=["a.jpg", "b.jpg", "c.jpg"],
        model_id="dinov2-base",
        normalized=True,
    )
    idx.save()
    loaded = EmbeddingIndex.load(tmp_path / "idx")
    assert loaded.model_id == "dinov2-base"
    assert len(loaded.image_paths) == 3
    assert loaded.embeddings.shape == (3, 768)


def test_search_index_top_k():
    from visionservex.runtime.embeddings import EmbeddingIndex, search_index

    # 3 unit vectors
    embs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    idx = EmbeddingIndex(
        index_dir=Path("/tmp/x"),
        embeddings=embs,
        image_paths=["a", "b", "c"],
        model_id="test",
    )
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    hits = search_index(idx, query, top_k=2)
    assert len(hits) == 2
    assert hits[0][0] == "a"  # closest match
    assert hits[0][1] > hits[1][1]


def test_deduplicate_index_finds_duplicates():
    from visionservex.runtime.embeddings import EmbeddingIndex, deduplicate_index

    # Two near-identical vectors and one distinct
    embs = np.array(
        [
            [1.0, 0.0],
            [0.9999, 0.0141],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    idx = EmbeddingIndex(
        index_dir=Path("/tmp/x"),
        embeddings=embs,
        image_paths=["a", "b", "c"],
        model_id="test",
    )
    pairs = deduplicate_index(idx, threshold=0.95)
    assert len(pairs) >= 1
    # The duplicate pair should be (a, b)
    pair_ids = {(p[0], p[1]) for p in pairs}
    assert ("a", "b") in pair_ids


# ============================================================
# CLI: embeddings (smoke with mock — but mock-detect doesn't return EmbeddingResult)
# ============================================================


def test_embed_cli_invalid_model():
    """mock-detect returns DetectionResult, not EmbeddingResult — should fail gracefully."""
    import tempfile

    from PIL import Image as _PIL

    with tempfile.TemporaryDirectory() as tmp:
        img = _PIL.new("RGB", (320, 240))
        img_path = Path(tmp) / "test.jpg"
        img.save(str(img_path))
        result = runner.invoke(app, ["embed", "mock-detect", str(img_path)])
        # Should fail because mock-detect doesn't return embeddings
        assert result.exit_code != 0


# ============================================================
# New task type
# ============================================================


def test_task_embed_in_literal():
    # Task is a Literal — check via repr or string match
    import typing

    from visionservex.registry.registry import Task

    args = typing.get_args(Task)
    assert "embed" in args
    assert "vlm" in args


def test_dinov2_in_registry():
    from visionservex.registry import default_registry

    reg = default_registry()
    entry = reg.get("dinov2-base")
    assert entry.task == "embed"
    assert entry.implementation_status == "wired"
    assert entry.model_category == "feature_backbone"
    assert entry.license == "Apache-2.0"


def test_siglip2_in_registry():
    from visionservex.registry import default_registry

    reg = default_registry()
    entry = reg.get("siglip2-base-patch16-224")
    assert entry.task == "embed"
    assert entry.implementation_status == "wired"


def test_florence_in_registry():
    from visionservex.registry import default_registry

    reg = default_registry()
    entry = reg.get("florence-2-base")
    assert entry.task == "vlm"
    # v1.8.0 wired florence-2-base via Florence2Engine.
    assert entry.implementation_status in {"wired", "partial"}
    assert entry.engine == "florence2"


def test_owlv2_in_registry():
    from visionservex.registry import default_registry

    reg = default_registry()
    entry = reg.get("owlv2-base-patch16")
    assert entry.task == "open_vocab_detect"
    assert entry.license == "Apache-2.0"


# ============================================================
# Version check
# ============================================================


def test_version_is_at_least_160():
    from visionservex import __version__

    # Version may bump in later releases; this test guards that 1.6.0 features
    # remain available. Accept 1.6.x and any 1.>=7 / 2.x.
    parts = __version__.split(".")
    assert int(parts[0]) >= 1
    assert int(parts[0]) > 1 or int(parts[1]) >= 6, (
        f"v1.6.0 features require >= 1.6.0; got {__version__}"
    )


# ============================================================
# Benchmark-embeddings command
# ============================================================


def test_benchmark_embeddings_missing_dataset():
    result = runner.invoke(
        app, ["benchmark-embeddings", "--model", "dinov2-small", "--dataset", "synthetic"]
    )
    # synthetic isn't folder: prefix
    assert result.exit_code != 0


def test_benchmark_embeddings_missing_path():
    result = runner.invoke(
        app,
        ["benchmark-embeddings", "--model", "dinov2-small", "--dataset", "folder:/nonexistent"],
    )
    assert result.exit_code != 0
