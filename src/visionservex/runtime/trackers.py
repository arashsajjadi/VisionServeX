# SPDX-License-Identifier: Apache-2.0
"""Tracker adapter registry for VisionServeX video-search.

All trackers expose:
    update(detections, frame_idx, timestamp_s) -> list[TrackBox]

where TrackBox is from simple_tracker.
"""

from __future__ import annotations

from typing import Any

_TRACKER_PACKAGES = {
    "bytetrack": "bytetracker",
    "bot-sort": "botsort",
    "ocsort": "ocsort",
    "deepsort": "deep_sort_realtime",
}


class TrackerUnavailableError(Exception):
    def __init__(self, name: str, code: str, install: str):
        super().__init__(f"{code}: {name}")
        self.name = name
        self.code = code
        self.install = install

    def to_dict(self) -> dict:
        return {"code": self.code, "tracker": self.name, "install": self.install}


def _probe(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def build_tracker(name: str, **kwargs: Any) -> Any:
    """Return a tracker instance for the given name.

    Raises TrackerUnavailableError if the required package is missing.
    Returns None for simple-iou so callers can use SimpleIoUTracker directly.
    """
    if name in ("simple-iou", "simple_iou", ""):
        return None  # caller uses SimpleIoUTracker

    pkg = _TRACKER_PACKAGES.get(name)
    if not pkg:
        raise TrackerUnavailableError(
            name,
            "TRACKER_UNKNOWN",
            f"Unknown tracker {name!r}. Available: simple-iou, bytetrack, bot-sort, ocsort",
        )

    if not _probe(pkg):
        install_map = {
            "bytetrack": "pip install bytetracker",
            "bot-sort": "git clone https://github.com/NirAharon/BoT-SORT && pip install -e .",
            "ocsort": "pip install ocsort",
            "deepsort": "pip install deep-sort-realtime",
        }
        raise TrackerUnavailableError(
            name,
            f"{name.upper().replace('-', '')}_REQUIRED".replace("BYTETRACK", "BYTETRACK"),
            install_map.get(name, f"pip install {pkg}"),
        )

    return _build_bytetrack_adapter(**kwargs) if name == "bytetrack" else None


class _ByteTrackAdapter:
    """Adapter wrapping bytetracker to VisionServeX TrackBox interface."""

    def __init__(self, **kwargs: Any) -> None:
        # bytetracker API: BYTETracker(args) or BYTETracker(frame_rate=...)
        from bytetracker import BYTETracker  # type: ignore

        class _Args:
            track_thresh = kwargs.get("track_thresh", 0.5)
            track_buffer = kwargs.get("track_buffer", 30)
            match_thresh = kwargs.get("match_thresh", 0.8)
            mot20 = False

        try:
            self._tracker = BYTETracker(_Args(), frame_rate=kwargs.get("frame_rate", 30))
        except TypeError:
            # Older bytetracker versions
            self._tracker = BYTETracker(_Args())
        self._frame_count = 0

    def update(
        self,
        detections: list[tuple[tuple[float, float, float, float], float, str]],
        *,
        frame_idx: int = 0,
        timestamp_s: float = 0.0,
    ) -> list[Any]:
        import numpy as np

        from visionservex.runtime.simple_tracker import TrackBox

        if not detections:
            self._frame_count += 1
            return []

        # Build [x1, y1, x2, y2, score] array
        dets_arr = np.array(
            [[b[0], b[1], b[2], b[3], s] for (b, s, _lbl) in detections],
            dtype=np.float32,
        )
        try:
            tracked = self._tracker.update(dets_arr, img_size=(1080, 1920), img_info=(1080, 1920))
        except TypeError:
            tracked = self._tracker.update(dets_arr)
        self._frame_count += 1

        results = []
        for t in tracked:
            try:
                tid = int(t.track_id)
                tlbr = t.tlbr  # [x1, y1, x2, y2]
                score = float(t.score) if hasattr(t, "score") else 0.9
            except AttributeError:
                continue
            results.append(
                TrackBox(
                    track_id=tid,
                    box=(float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])),
                    score=score,
                    label="person",
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                )
            )
        return results


def _build_bytetrack_adapter(**kwargs: Any) -> _ByteTrackAdapter:
    return _ByteTrackAdapter(**kwargs)


__all__ = [
    "TrackerUnavailableError",
    "build_tracker",
]
