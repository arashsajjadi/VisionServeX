# SPDX-License-Identifier: Apache-2.0
"""v3.22.0 — video pipeline, tiling, ffmpeg tools, and job cancellation.

CI-safe: uses a tiny synthetic video + mock-detect (no torch/GPU). ffmpeg-
dependent assertions skip gracefully when ffmpeg is absent.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from visionservex.core.results import Box, Detection, DetectionResult
from visionservex.runtime.ffmpeg_tools import (
    BROWSER_PRESETS,
    VideoLimitError,
    VideoProbe,
    detect_hwaccel,
    enforce_limits,
)
from visionservex.runtime.jobs import get_job_store
from visionservex.runtime.tiling import _tile_grid, frame_diagnostics
from visionservex.runtime.video_pipeline import CancelToken


# --------------------------- tiling ---------------------------
def test_tile_grid_covers_image_with_overlap() -> None:
    wins = _tile_grid(1280, 720, tile=512, overlap=0.25)
    assert wins, "must produce tiles"
    # every tile within bounds
    for x1, y1, x2, y2 in wins:
        assert 0 <= x1 < x2 <= 1280
        assert 0 <= y1 < y2 <= 720
    # right/bottom edges covered
    assert max(x2 for _, _, x2, _ in wins) == 1280
    assert max(y2 for _, _, _, y2 in wins) == 720


def test_frame_diagnostics_empty_reason() -> None:
    empty = DetectionResult(kind="detection", model_id="m", detections=[])
    diag = frame_diagnostics(empty, raw_candidates=300, threshold=0.5)
    assert diag["final_count"] == 0
    assert diag["empty_reason"] and "300" in diag["empty_reason"]


def test_frame_diagnostics_counts_and_sizes() -> None:
    dets = [
        Detection(box=Box(0, 0, 10, 10), score=0.9, label="car", class_id=1),
        Detection(box=Box(0, 0, 5, 5), score=0.5, label="car", class_id=1),
        Detection(box=Box(0, 0, 20, 20), score=0.7, label="bus", class_id=2),
    ]
    diag = frame_diagnostics(DetectionResult(kind="detection", detections=dets), threshold=0.3)
    assert diag["final_count"] == 3
    assert diag["class_distribution"] == {"car": 2, "bus": 1}
    assert diag["size_min_px"] == 25.0
    assert diag["empty_reason"] is None


# --------------------------- ffmpeg tools ---------------------------
def test_browser_presets_exist() -> None:
    assert {"480p", "720p", "1080p", "source"} <= set(BROWSER_PRESETS)


def test_detect_hwaccel_returns_dict() -> None:
    hw = detect_hwaccel()
    assert {"ffmpeg", "nvenc", "nvenc_available", "nvdec_available"} <= set(hw)


def test_enforce_limits_raises_on_oversize() -> None:
    p = VideoProbe(path="x", exists=True, size_bytes=10**9, duration_s=10, width=100, height=100)
    with pytest.raises(VideoLimitError):
        enforce_limits(p, max_bytes=1000)


def test_enforce_limits_raises_on_long_duration() -> None:
    p = VideoProbe(path="x", exists=True, size_bytes=10, duration_s=10000, width=10, height=10)
    with pytest.raises(VideoLimitError):
        enforce_limits(p, max_duration_s=60)


# --------------------------- job cancellation ---------------------------
def test_job_cancel_sets_event_and_flag() -> None:
    store = get_job_store()
    job = store.create(model_id="mock-detect", kind="video_infer")
    assert job.cancel_requested is False
    assert store.cancel(job.job_id) is True
    refreshed = store.get(job.job_id)
    assert refreshed.cancel_event.is_set() is True
    assert refreshed.cancel_requested is True
    assert refreshed.status == "cancelled"
    assert refreshed.to_dict()["cancel_requested"] is True


def test_cancel_token_wraps_job_event() -> None:
    store = get_job_store()
    job = store.create(model_id="m", kind="video_infer")
    token = CancelToken(_event=job.cancel_event)
    assert token.cancelled is False
    store.cancel(job.job_id)
    assert token.cancelled is True


# --------------------------- video inference (synthetic + mock) ---------------------------
def _make_video(path: str, frames: int = 12):
    from visionservex.runtime.video_io import make_synthetic_video

    return make_synthetic_video(path, frames=frames, width=128, height=96, fps=12.0)


def test_infer_video_streams_and_preserves_frame_identity() -> None:
    from visionservex.core.model import VisionModel
    from visionservex.runtime.video_pipeline import infer_video

    with tempfile.TemporaryDirectory() as td:
        out = _make_video(str(Path(td) / "v.mp4"), frames=12)
        vm = VisionModel("mock-detect", device="cpu")
        report = infer_video(vm, str(out), max_frames=12, mode="balanced")
        assert report["frames_processed"] > 0
        idxs = [f["frame_index"] for f in report["frames"]]
        assert len(set(idxs)) == len(idxs), "frame indices must be unique (independence)"
        ts = [f["time_sec"] for f in report["frames"]]
        assert ts == sorted(ts), "timestamps must be monotonic"
        assert report["cancelled"] is False
        assert "bottleneck_summary" in report


def test_infer_video_cancellation_stops_early() -> None:
    from visionservex.core.model import VisionModel
    from visionservex.runtime.video_pipeline import infer_video

    with tempfile.TemporaryDirectory() as td:
        out = _make_video(str(Path(td) / "v.mp4"), frames=30)
        vm = VisionModel("mock-detect", device="cpu")
        token = CancelToken()
        token.cancel()  # cancel before starting → no waves should run
        report = infer_video(vm, str(out), max_frames=30, cancel=token)
        assert report["cancelled"] is True
        assert report["frames_processed"] == 0
