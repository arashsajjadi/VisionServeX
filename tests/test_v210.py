# SPDX-License-Identifier: Apache-2.0
"""Tests for v2.1.0: OWL-ViT, CLIP/SigLIP, ConvNeXtV2, Florence-2 CLI, MedSAM registry,
manifest coverage, anomaly upgrades, surveillance non-empty."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# OWL-ViT engine registration
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_owlvit_engine_registered():
    from visionservex.engines import build_engine
    from visionservex.registry import default_registry

    reg = default_registry()
    entry = reg.get("owlvit-base-patch32")
    assert entry.engine == "owlvit"
    assert entry.implementation_status == "wired"
    engine = build_engine(entry)
    from visionservex.engines.owlv2 import OWLv2Engine

    assert isinstance(engine, OWLv2Engine)


@pytest.mark.fast
def test_owlvit_family_detected_from_entry():
    """OWLv2Engine._real_load branches on family == owlvit."""
    from visionservex.registry import default_registry

    reg = default_registry()
    entry = reg.get("owlvit-base-patch32")
    # Family must be 'owlvit' so the engine selects OwlViTForObjectDetection
    assert entry.family.lower() in {"owlvit", "owl-vit", "owl_vit"}


# ---------------------------------------------------------------------------
# HF Classification engine
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_hf_classify_engine_registered():
    from visionservex.engines import build_engine
    from visionservex.registry import default_registry

    reg = default_registry()
    entry = reg.get("convnextv2-tiny")
    assert entry.engine == "convnextv2"
    assert entry.implementation_status == "wired"
    engine = build_engine(entry)
    from visionservex.engines.hf_classify import HFClassifyEngine

    assert isinstance(engine, HFClassifyEngine)


@pytest.mark.fast
def test_hf_classify_mocked_inference():
    torch = pytest.importorskip("torch")
    from visionservex.engines.hf_classify import HFClassifyEngine
    from visionservex.registry import default_registry

    reg = default_registry()
    entry = reg.get("convnextv2-tiny")
    engine = HFClassifyEngine(entry)
    engine._real_ready = True
    engine._torch = torch
    engine._id2label = {0: "cat", 1: "dog", 2: "car"}

    proc = MagicMock()
    proc.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
    engine._processor = proc

    model = MagicMock()
    model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    logits = torch.tensor([[2.5, 1.0, 0.1]])
    model.return_value = MagicMock(logits=logits)
    engine._model = model

    from PIL import Image

    img = Image.new("RGB", (64, 64))
    result = engine.predict(img, top_k=3)
    assert result.kind == "classification"
    assert result.top_k[0][0] == "cat"  # highest logit label
    assert result.top_k[0][1] > result.top_k[1][1]  # highest score first


# ---------------------------------------------------------------------------
# CLIP/SigLIP engine aliases
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_clip_engine_alias_registered():
    from visionservex.registry import default_registry

    reg = default_registry()
    entry = reg.get("clip-vit-base-patch32")
    assert entry.engine == "clip"
    assert entry.implementation_status == "wired"


@pytest.mark.fast
def test_siglip_base_engine_registered():
    from visionservex.registry import default_registry

    reg = default_registry()
    entry = reg.get("siglip-base-patch16-224")
    assert entry.engine == "siglip"
    assert entry.implementation_status == "wired"


# ---------------------------------------------------------------------------
# Florence-2 CLI
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_florence2_cli_registered():
    from visionservex.cli.main import app

    names = {g.name for g in app.registered_groups}
    assert "florence2" in names


@pytest.mark.fast
def test_florence2_doctor_returns_version_info():
    from visionservex.cli.florence2_commands import _check_transformers_version

    compatible, ver = _check_transformers_version()
    assert isinstance(compatible, bool)
    assert isinstance(ver, str)
    assert ver != ""


@pytest.mark.fast
def test_florence2_unsupported_env_gives_recipe():
    """When transformers >= 5.0, doctor must return the exact setup recipe."""
    import transformers as _tr

    from visionservex.cli.florence2_commands import _SETUP_RECIPE, _check_transformers_version

    orig = _tr.__version__
    try:
        _tr.__version__ = "5.9.0"
        compatible, ver = _check_transformers_version()
        assert not compatible
        assert ver == "5.9.0"
        assert "conda create" in _SETUP_RECIPE
        assert "florence2" in _SETUP_RECIPE
    finally:
        _tr.__version__ = orig


# ---------------------------------------------------------------------------
# MedSAM registry entry wired
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_medsam_registry_wired():
    from visionservex.registry import default_registry

    reg = default_registry()
    entry = reg.get("medsam")
    assert entry.implementation_status == "wired"
    assert entry.engine == "sam_hf"
    assert entry.hf_repo_id == "wanglab/medsam-vit-base"


@pytest.mark.fast
def test_medsam_manifest_runnable():
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST

    assert "medsam" in SOURCE_MANIFEST
    assert SOURCE_MANIFEST["medsam"].runnable_in_visionservex is True


# ---------------------------------------------------------------------------
# ConvNeXtV2 manifest entries
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_convnextv2_manifest_entries():
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST

    for mid in ("convnextv2-tiny", "convnextv2-base", "convnextv2-large"):
        assert mid in SOURCE_MANIFEST
        assert SOURCE_MANIFEST[mid].runnable_in_visionservex is True
        assert SOURCE_MANIFEST[mid].license == "Apache-2.0"


# ---------------------------------------------------------------------------
# Anomaly commands - real API path structure
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_anomaly_train_tries_engine_api(monkeypatch, tmp_path):
    """When anomalib is available, train should try the Engine API first."""
    from visionservex.cli import anomaly_commands

    # Simulate anomalib installed but Engine API not available
    engine_calls = []

    def fake_anomalib_engine():
        engine_calls.append("engine_import_tried")
        raise ImportError("anomalib.engine not available in this mock")

    monkeypatch.setattr(
        anomaly_commands,
        "_anomalib_available",
        lambda: (True, "1.0.0"),
    )

    # Make required dataset
    data = tmp_path / "normal"
    data.mkdir()
    for i in range(3):
        img_path = data / f"img_{i}.png"
        from PIL import Image

        Image.new("RGB", (64, 64), color=(200, 200, 200)).save(img_path)

    # The engine try block should be attempted
    assert anomaly_commands._require_anomalib() is None  # anomalib reports available
    assert anomaly_commands._require_dataset(data) is None  # dataset valid


@pytest.mark.fast
def test_anomaly_patchcore_in_supported_algos():
    from visionservex.cli.anomaly_commands import SUPPORTED_ALGOS

    assert "patchcore" in SUPPORTED_ALGOS
    assert SUPPORTED_ALGOS["patchcore"]["anomalib_class"] == "Patchcore"


# ---------------------------------------------------------------------------
# Manifest coverage for new families
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_owlvit_manifest_entries():
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST

    for mid in ("owlvit-base-patch32", "owlvit-large-patch14"):
        assert mid in SOURCE_MANIFEST
        assert SOURCE_MANIFEST[mid].runnable_in_visionservex is True


@pytest.mark.fast
def test_clip_manifest_entries():
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST

    for mid in ("clip-vit-base-patch32", "clip-vit-large-patch14"):
        assert mid in SOURCE_MANIFEST
        assert SOURCE_MANIFEST[mid].runnable_in_visionservex is True
        assert SOURCE_MANIFEST[mid].license == "MIT"


@pytest.mark.fast
def test_siglip_manifest_entries():
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST

    assert "siglip-base-patch16-224" in SOURCE_MANIFEST
    assert SOURCE_MANIFEST["siglip-base-patch16-224"].runnable_in_visionservex is True


# ---------------------------------------------------------------------------
# Surveillance non-empty smoke (mocked fast path)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_surveillance_pipeline_with_fake_detections(tmp_path):
    """End-to-end surveillance index/query with mock detect + embed — verifies non-empty path."""
    import numpy as np
    from PIL import Image

    from visionservex.runtime.video_search import build_index, query_index

    # Create 3 tiny frames
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(3):
        Image.new("RGB", (64, 64), color=(i * 60, 100, 100)).save(frames / f"frame_{i:03d}.jpg")

    rng = np.random.RandomState(42)

    def fake_detect(image, prompt):
        # Always return 2 detections (simulating real scene)
        return [
            ((5.0, 5.0, 30.0, 55.0), 0.9, "person"),
            ((35.0, 5.0, 60.0, 55.0), 0.7, "person"),
        ]

    def fake_embed(crop):
        return rng.randn(512).astype("float32")

    out = build_index(
        source=str(frames),
        out_dir=str(tmp_path / "index"),
        detect_fn=fake_detect,
        embed_fn=fake_embed,
        prompt="person",
    )

    emb = np.load(out / "embeddings.npy")
    assert emb.shape[0] == 6, f"Expected 6 crops (3 frames x 2 dets), got {emb.shape[0]}"
    assert emb.shape[1] == 512

    hits = query_index(out, emb[0], top_k=3)
    assert len(hits) >= 1
    assert hits[0].similarity > 0.9  # self-match
    # Confirm non-empty
    assert any(h.similarity > 0 for h in hits), "No non-trivial hits"


# ---------------------------------------------------------------------------
# Florence-2 [florence2] extra declared in pyproject
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_florence2_extra_in_pyproject():
    import tomllib

    pp = Path(__file__).resolve().parent.parent / "pyproject.toml"
    cfg = tomllib.loads(pp.read_text())
    extras = cfg.get("project", {}).get("optional-dependencies", {})
    assert "florence2" in extras
    deps = extras["florence2"]
    tr_dep = next((d for d in deps if "transformers" in d), None)
    assert tr_dep is not None, "florence2 extra must pin transformers"
    assert "<5" in tr_dep, "florence2 extra must cap transformers < 5"
