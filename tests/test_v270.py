# SPDX-License-Identifier: Apache-2.0
"""Tests for v2.7.0: real-execution paths.

These tests verify the SHAPE of the real-execution paths (tracker-smoke,
anomalib adapter, OSNet adapter, sidecar scripts). The actual heavy
installs (anomalib, openmmlab, detectron2) live behind opt-in env vars.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# tracker-smoke CLI command
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_tracker_smoke_simple_iou_synthetic_sequence():
    from visionservex.cli.video_search_commands import app as vs_app

    runner = CliRunner()
    result = runner.invoke(vs_app, ["tracker-smoke", "--tracker", "simple-iou", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "OK"
    assert payload["tracker"] == "simple-iou"
    assert payload["frames"] == 3
    assert payload["n_tracks"] >= 1
    assert all("track_id" in row for row in payload["tracks"])


@pytest.mark.fast
def test_tracker_smoke_bytetrack_missing_returns_blocker():
    from visionservex.cli.video_search_commands import app as vs_app

    runner = CliRunner()
    result = runner.invoke(vs_app, ["tracker-smoke", "--tracker", "bytetrack", "--json"])
    # bytetrack is not installed in this environment.
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["code"] in {"BYTETRACK_REQUIRED", "OK"}


@pytest.mark.fast
def test_tracker_smoke_writes_out_file(tmp_path):
    from visionservex.cli.video_search_commands import app as vs_app

    out = tmp_path / "tracks.json"
    runner = CliRunner()
    result = runner.invoke(
        vs_app,
        ["tracker-smoke", "--tracker", "simple-iou", "--out", str(out), "--json"],
    )
    assert result.exit_code == 0
    assert out.exists()
    saved = json.loads(out.read_text())
    assert saved["code"] == "OK"
    assert saved["tracker"] == "simple-iou"


# ---------------------------------------------------------------------------
# Anomalib adapter — version dispatch + helpers
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_latest_checkpoint_picks_newest(tmp_path):
    from visionservex.integrations.anomalib_adapter import PatchCoreAdapter

    a = tmp_path / "a.ckpt"
    b = tmp_path / "b.ckpt"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    # Bump b's mtime so it's newer.
    import os
    import time

    os.utime(b, (time.time() + 10, time.time() + 10))
    assert PatchCoreAdapter._latest_checkpoint(tmp_path).name == "b.ckpt"


@pytest.mark.fast
def test_latest_checkpoint_returns_none_when_empty(tmp_path):
    from visionservex.integrations.anomalib_adapter import PatchCoreAdapter

    assert PatchCoreAdapter._latest_checkpoint(tmp_path) is None


@pytest.mark.fast
def test_anomalib_adapter_2x_folder_signature(monkeypatch, tmp_path):
    """When anomalib 2.x Folder requires name=, the adapter must pass it."""
    fake_anomalib = MagicMock()
    fake_anomalib.__version__ = "2.4.2"

    folder_calls: list[dict] = []

    class _FakeFolder:
        def __init__(self, **kwargs):
            if "name" not in kwargs:
                raise TypeError("missing required 'name'")
            folder_calls.append(kwargs)

    class _FakePatchcore:
        def __init__(self):
            pass

    class _FakeEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, **kwargs):
            pass

    fake_anomalib.engine.Engine = _FakeEngine
    fake_anomalib.models.Patchcore = _FakePatchcore
    fake_anomalib.data.Folder = _FakeFolder

    monkeypatch.setitem(sys.modules, "anomalib", fake_anomalib)
    monkeypatch.setitem(sys.modules, "anomalib.engine", fake_anomalib.engine)
    monkeypatch.setitem(sys.modules, "anomalib.models", fake_anomalib.models)
    monkeypatch.setitem(sys.modules, "anomalib.data", fake_anomalib.data)

    from visionservex.integrations.anomalib_adapter import PatchCoreAdapter

    data = tmp_path / "normal"
    data.mkdir()
    out = tmp_path / "out"
    adapter = PatchCoreAdapter()
    result = adapter.train(data_dir=data, out_dir=out)

    assert result["status"] == "trained"
    assert folder_calls, "Folder was not constructed"
    assert folder_calls[0]["name"] == "visionservex-anomaly"
    assert folder_calls[0]["normal_dir"] == data.name
    assert Path(folder_calls[0]["root"]).resolve() == data.parent.resolve()


# ---------------------------------------------------------------------------
# Real-package adapters — bytetracker / ocsort / torchreid
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_bytetrack_real_smoke_when_installed():
    pytest.importorskip("bytetracker")
    pytest.importorskip("torch")

    from visionservex.runtime.trackers import build_tracker

    tracker = build_tracker("bytetrack")
    dets = [
        ((10.0, 20.0, 100.0, 200.0), 0.9, "person"),
        ((30.0, 40.0, 200.0, 300.0), 0.8, "person"),
    ]
    out = tracker.update(dets, frame_idx=0, timestamp_s=0.0)
    assert out, "expected at least one track from real bytetracker"
    assert all(getattr(tb, "track_id", None) is not None for tb in out)


@pytest.mark.fast
def test_ocsort_real_smoke_when_installed():
    pytest.importorskip("ocsort")
    pytest.importorskip("torch")
    pytest.importorskip("filterpy")

    from visionservex.runtime.trackers import build_tracker

    tracker = build_tracker("ocsort")
    dets = [
        ((10.0, 20.0, 100.0, 200.0), 0.9, "person"),
        ((30.0, 40.0, 200.0, 300.0), 0.8, "person"),
    ]
    out = tracker.update(dets, frame_idx=0, timestamp_s=0.0)
    assert out, "expected at least one track from real ocsort"


@pytest.mark.fast
def test_torchreid_dual_path_layout_lookup(monkeypatch):
    """torchreid 0.2.5 lives at torchreid.reid.utils; legacy at torchreid.utils."""
    legacy = MagicMock()

    class _FakeFE:
        def __init__(self, **_):
            pass

    legacy.FeatureExtractor = _FakeFE
    fake_top = MagicMock()
    fake_top.reid = MagicMock()
    fake_top.reid.utils = legacy
    # Intentionally make torchreid.utils miss FeatureExtractor.
    fake_top.utils = MagicMock(spec=[])

    monkeypatch.setitem(sys.modules, "torchreid", fake_top)
    monkeypatch.setitem(sys.modules, "torchreid.utils", fake_top.utils)
    monkeypatch.setitem(sys.modules, "torchreid.reid", fake_top.reid)
    monkeypatch.setitem(sys.modules, "torchreid.reid.utils", legacy)

    from visionservex.runtime.reid import build_reid_extractor

    ckpt = Path("/tmp/_fake_osnet_ckpt_test_v270.pth")
    ckpt.write_bytes(b"x")
    try:
        adapter = build_reid_extractor("osnet", model_path=str(ckpt))
        # Reached the FeatureExtractor; if it raises that's an API mismatch only
        # but layout dispatch already succeeded.
        assert adapter is not None
    finally:
        ckpt.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Sidecar scripts ship and are executable
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_anomaly_sidecar_script_exists_and_executable():
    p = ROOT / "scripts" / "run_anomaly_smoke.sh"
    assert p.exists()
    assert p.stat().st_mode & 0o100, "run_anomaly_smoke.sh must be executable"


@pytest.mark.fast
def test_openmmlab_sidecar_script_exists_and_executable():
    p = ROOT / "scripts" / "run_openmmlab_smoke.sh"
    assert p.exists()
    assert p.stat().st_mode & 0o100


@pytest.mark.fast
def test_maskdino_sidecar_script_refuses_missing_checkpoint(tmp_path):
    """The MaskDINO sidecar must NOT invent a checkpoint URL."""
    import subprocess

    p = ROOT / "scripts" / "run_maskdino_smoke.sh"
    assert p.exists()
    img = tmp_path / "empty.jpg"
    img.write_bytes(b"x")
    res = subprocess.run(
        ["bash", str(p), str(img)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert res.returncode != 0
    assert "CHECKPOINT_REQUIRED" in res.stdout or "CONDA_REQUIRED" in res.stdout


@pytest.mark.fast
def test_anomaly_fixture_present_for_smoke_script():
    normal = ROOT / "tests" / "fixtures" / "anomaly" / "normal"
    assert normal.exists()
    images = [p for p in normal.iterdir() if p.suffix in {".png", ".jpg"}]
    assert len(images) >= 8, "anomaly smoke needs ≥ 8 normal images"


# ---------------------------------------------------------------------------
# Real HF smoke gate — opt-in
# ---------------------------------------------------------------------------


@pytest.mark.real_model
def test_dinov2_real_embed():
    """Real HF DINOv2 embedding — opt-in via VISIONSERVEX_RUN_REAL_MODEL_TESTS=1."""
    import os

    if not os.environ.get("VISIONSERVEX_RUN_REAL_MODEL_TESTS"):
        pytest.skip("set VISIONSERVEX_RUN_REAL_MODEL_TESTS=1 to enable")

    from PIL import Image

    from visionservex import VisionModel

    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    model = VisionModel("dinov2-base", auto_pull=False)
    result = model.predict(img)
    assert result is not None


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_version_is_at_least_270():
    import visionservex

    parts = tuple(int(p) for p in visionservex.__version__.split(".")[:3])
    assert parts >= (2, 7, 0), f"version {visionservex.__version__} < 2.7.0"
