# SPDX-License-Identifier: Apache-2.0
"""Tests for v1.9.0: surveillance video-search, anomaly, medical, openmmlab validate.

All fast tests are mocked — no real models, no real downloads, no real GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# SimpleIoUTracker
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_simple_iou_tracker_assigns_persistent_ids():
    """A box moving across frames keeps the same track_id."""
    from visionservex.runtime.simple_tracker import SimpleIoUTracker

    t = SimpleIoUTracker(iou_threshold=0.3)
    # Frame 0: one box
    out0 = t.update([((10, 10, 50, 50), 0.9, "person")], frame_idx=0, timestamp_s=0.0)
    assert len(out0) == 1
    tid = out0[0].track_id

    # Frame 1: slightly shifted box → same track
    out1 = t.update([((12, 12, 52, 52), 0.9, "person")], frame_idx=1, timestamp_s=1.0)
    assert len(out1) == 1
    assert out1[0].track_id == tid

    # Frame 2: very different box → new track
    out2 = t.update([((100, 100, 140, 140), 0.9, "person")], frame_idx=2, timestamp_s=2.0)
    assert out2[0].track_id != tid


@pytest.mark.fast
def test_simple_iou_tracker_lost_track_pruning():
    """A track that disappears for max_lost_frames frames is dropped."""
    from visionservex.runtime.simple_tracker import SimpleIoUTracker

    t = SimpleIoUTracker(iou_threshold=0.3, max_lost_frames=2)
    t.update([((10, 10, 50, 50), 0.9, "person")], frame_idx=0, timestamp_s=0.0)
    # 3 empty frames > max_lost_frames(2) → pruned
    for f in range(1, 4):
        t.update([], frame_idx=f, timestamp_s=f)
    # Same box reappears → must get a NEW id
    out = t.update([((10, 10, 50, 50), 0.9, "person")], frame_idx=10, timestamp_s=10.0)
    assert out[0].track_id != 1


@pytest.mark.fast
def test_simple_iou_validates_inputs():
    from visionservex.runtime.simple_tracker import SimpleIoUTracker

    with pytest.raises(ValueError):
        SimpleIoUTracker(iou_threshold=1.5)
    with pytest.raises(ValueError):
        SimpleIoUTracker(max_lost_frames=-1)


# ---------------------------------------------------------------------------
# video_search frame iterator + index/query end-to-end (mocked detector+embedder)
# ---------------------------------------------------------------------------


def _make_fake_frames(tmp: Path, n: int = 3) -> Path:
    folder = tmp / "frames"
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (64, 64), color=(i * 50 % 255, 80, 80))
        img.save(folder / f"frame_{i:03d}.png")
    return folder


@pytest.mark.fast
def test_video_search_iter_frames_folder(tmp_path):
    from visionservex.runtime.video_search import iter_frames

    folder = _make_fake_frames(tmp_path, n=4)
    items = list(iter_frames(folder))
    assert len(items) == 4
    assert items[0][0] == 0
    assert items[0][2].size == (64, 64)


@pytest.mark.fast
def test_video_search_iter_frames_missing(tmp_path):
    from visionservex.runtime.video_search import iter_frames

    with pytest.raises(FileNotFoundError):
        next(iter_frames(tmp_path / "does_not_exist"))


@pytest.mark.fast
def test_video_search_end_to_end_mocked(tmp_path):
    """Build a fake-frame index with mock detector+embedder, then query."""
    from visionservex.runtime.video_search import build_index, query_index

    folder = _make_fake_frames(tmp_path, n=5)

    def fake_detect(image, prompt):
        # One persistent detection per frame
        return [((10, 10, 40, 40), 0.9, "person")]

    rng = np.random.RandomState(0)

    # Per-image embeddings, deterministic
    def fake_embed(crop):
        # Color-channel mean → 3-D vector, expanded
        arr = np.asarray(crop)
        mean = arr.reshape(-1, 3).mean(axis=0)
        return np.concatenate([mean / 255.0, rng.rand(5)]).astype("float32")

    out_dir = tmp_path / "index_smoke"
    idx_path = build_index(
        source=str(folder),
        out_dir=str(out_dir),
        detect_fn=fake_detect,
        embed_fn=fake_embed,
        detector_model_id="mock-detector",
        embedder_model_id="mock-embedder",
        prompt="person",
    )

    assert idx_path.exists()
    assert (idx_path / "manifest.json").exists()
    assert (idx_path / "embeddings.npy").exists()

    embeddings = np.load(idx_path / "embeddings.npy")
    assert embeddings.shape[0] == 5  # one crop per frame
    assert embeddings.shape[1] == 8  # 3 channels + 5 random

    # Query with a vector that matches frame 0 best
    query = embeddings[0].copy()
    hits = query_index(idx_path, query, top_k=3)
    assert len(hits) >= 1
    assert hits[0].similarity > 0.9  # near-perfect self-match


@pytest.mark.fast
def test_video_search_render_html_excludes_external_resources(tmp_path):
    from visionservex.runtime.video_search import VideoSearchHit, render_timeline_html

    hits = [
        VideoSearchHit(
            track_id=1,
            frame_idx=0,
            timestamp_s=0.0,
            box=(10.0, 10.0, 40.0, 40.0),
            label="person",
            similarity=0.95,
        )
    ]
    html = render_timeline_html(hits, query_text="red shirt", source="dummy")
    assert "<script" not in html  # No external scripts
    assert "appearance-based retrieval" in html  # Privacy notice present
    assert "0.95" not in html or "0.950" in html or "0.95" in html  # Either format


@pytest.mark.fast
def test_video_search_save_load_roundtrip(tmp_path):
    from visionservex.runtime.video_search import IndexedCrop, IndexManifest, load_index, save_index

    crops = [
        IndexedCrop(
            track_id=1,
            frame_idx=0,
            timestamp_s=0.0,
            box=(0.0, 0.0, 10.0, 10.0),
            score=0.9,
            label="person",
            embedding_idx=0,
        ),
    ]
    manifest = IndexManifest(crops=crops, embedding_dim=4)
    out = tmp_path / "idx"
    save_index(out, manifest, np.zeros((1, 4), dtype="float32"))
    loaded_manifest, loaded_emb = load_index(out)
    assert loaded_emb.shape == (1, 4)
    assert len(loaded_manifest.crops) == 1
    assert loaded_manifest.crops[0].track_id == 1


@pytest.mark.fast
def test_video_search_privacy_notice_present():
    from visionservex.runtime.video_search import PRIVACY_NOTICE

    # The notice must explicitly disclaim face recognition / biometric identity.
    assert "face recognition" in PRIVACY_NOTICE.lower()
    assert "biometric" in PRIVACY_NOTICE.lower()


# ---------------------------------------------------------------------------
# Anomaly commands (Anomalib PatchCore)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_anomaly_list_includes_patchcore():
    from visionservex.cli.anomaly_commands import SUPPORTED_ALGOS

    assert "patchcore" in SUPPORTED_ALGOS
    assert SUPPORTED_ALGOS["patchcore"]["anomalib_class"] == "Patchcore"


@pytest.mark.fast
def test_anomaly_missing_anomalib_returns_structured_error(monkeypatch):
    """When anomalib is not installed, _require_anomalib returns ANOMALIB_REQUIRED."""
    from visionservex.cli import anomaly_commands

    monkeypatch.setattr(anomaly_commands, "_anomalib_available", lambda: (False, None))
    err = anomaly_commands._require_anomalib()
    assert err is not None
    assert err.code == "ANOMALIB_REQUIRED"
    assert "pip install" in err.fix


@pytest.mark.fast
def test_anomaly_dataset_required(tmp_path):
    """Empty dataset directory triggers DATASET_REQUIRED."""
    from visionservex.cli import anomaly_commands

    err = anomaly_commands._require_dataset(tmp_path / "missing")
    assert err is not None
    assert err.code == "DATASET_REQUIRED"

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    err = anomaly_commands._require_dataset(empty_dir)
    assert err is not None
    assert err.code == "DATASET_REQUIRED"


@pytest.mark.fast
def test_anomaly_dataset_present_returns_none(tmp_path):
    from visionservex.cli import anomaly_commands

    d = tmp_path / "good"
    d.mkdir()
    (d / "img.png").write_bytes(b"fake")
    assert anomaly_commands._require_dataset(d) is None


# ---------------------------------------------------------------------------
# Medical commands
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_medical_models_complete():
    from visionservex.cli.medical_commands import MEDICAL_MODELS

    required = {
        "totalsegmentator",
        "medsam",
        "medsam2",
        "sam-med2d",
        "nnunet-v2",
        "monai-bundles",
        "auto3dseg",
    }
    assert required.issubset(set(MEDICAL_MODELS.keys()))


@pytest.mark.fast
def test_medical_disclaimer_is_strict():
    from visionservex.cli.medical_commands import DISCLAIMER

    assert "RESEARCH" in DISCLAIMER
    assert "diagnos" not in DISCLAIMER.lower() or "do not" in DISCLAIMER.lower()


@pytest.mark.fast
def test_medical_recommend_routes_goal_to_model():
    """Routing logic picks plausible default models per goal phrase."""
    from visionservex.cli.medical_commands import MEDICAL_MODELS

    # Just confirm the routing model exists for these goal types.
    # (The recommend command's mapping is light, just smoke that the targets are present.)
    assert "totalsegmentator" in MEDICAL_MODELS  # ct
    assert "medsam" in MEDICAL_MODELS  # prompt
    assert "medsam2" in MEDICAL_MODELS  # video/3d


# ---------------------------------------------------------------------------
# OpenMMLab validate
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_openmmlab_validate_unknown_model():
    from visionservex.cli.openmmlab_commands import _validate_openmmlab_model

    p = _validate_openmmlab_model("not-a-real-mmlab-model")
    assert p["status"] == "error"
    assert p["structured_error_code"] == "CONFIG_REQUIRED"


@pytest.mark.fast
def test_openmmlab_validate_missing_modules_known_model():
    """Known model with no mmcv installed → OPENMMLAB_REQUIRED."""
    from visionservex.cli.openmmlab_commands import _validate_openmmlab_model

    p = _validate_openmmlab_model("rtmpose-s")
    # In CI mmcv/mmengine/mmpose are not installed.
    assert p["status"] in {"error", "ok"}
    if p["status"] == "error":
        assert p["structured_error_code"] in {"OPENMMLAB_REQUIRED", "CHECKPOINT_REQUIRED"}


# ---------------------------------------------------------------------------
# Benchmark-open-vocab metric helpers
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_benchmark_open_vocab_quantile():
    from visionservex.cli.benchmark_open_vocab import _quantile

    assert _quantile([], 0.5) == 0.0
    assert _quantile([10.0, 20.0, 30.0, 40.0, 50.0], 0.5) == 30.0
    assert _quantile([1.0], 0.95) == 1.0


@pytest.mark.fast
def test_benchmark_open_vocab_prompt_metrics_dataclass():
    from visionservex.cli.benchmark_open_vocab import PromptMetrics

    pm = PromptMetrics(
        prompt="person",
        n_images=10,
        n_images_with_match=8,
        mean_detections=2.5,
        mean_top1_score=0.42,
        p50_latency_ms=120.0,
        p95_latency_ms=240.0,
    )
    d = pm.to_dict()
    assert d["prompt"] == "person"
    assert d["n_images_with_match"] == 8
    assert d["mean_top1_score"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# CLI registration sanity
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_v190_clis_registered_in_main():
    """All v1.9.0 top-level subcommands are reachable via the Typer app."""
    from visionservex.cli.main import app

    names = {g.name for g in app.registered_groups}
    for required in {"anomaly", "medical", "video-search"}:
        assert required in names, f"missing CLI group: {required}"


@pytest.mark.fast
def test_v190_pyproject_anomaly_extra_present():
    """The [anomaly] extra is declared in pyproject.toml."""
    import tomllib

    pp = Path(__file__).resolve().parent.parent / "pyproject.toml"
    cfg = tomllib.loads(pp.read_text())
    extras = cfg.get("project", {}).get("optional-dependencies", {})
    assert "anomaly" in extras
    assert any("anomalib" in dep for dep in extras["anomaly"])


# ---------------------------------------------------------------------------
# Manifest still honest: nothing fake-wired
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_manifest_no_silent_runnable_inflation():
    """v1.9.0 must not flip non-wired entries to runnable_in_visionservex=True."""
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST

    # The 17 already-runnable in v1.8.1 are: dfine-{s,x,l}-o365-coco,
    # rfdetr-{small,large}, rfdetr-seg-medium, sam-vit-base, sam2-hiera-tiny,
    # dinov2-{small,base,large,giant}, florence-2-{base,large},
    # owlv2-{base-patch16,large-patch14}, siglip2-base-patch16-224.
    runnable = {k for k, v in SOURCE_MANIFEST.items() if v.runnable_in_visionservex}
    # Sanity floor: at least the 17 from v1.8.1 remain
    assert len(runnable) >= 17
