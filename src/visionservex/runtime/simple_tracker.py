# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Simple IoU tracker.

A minimal multi-object tracker that associates detections across frames using
Intersection-over-Union (IoU) between boxes. Suitable for the v1.9 surveillance
pipeline as a fallback when ByteTrack/OC-SORT are not available.

NOT a research-grade tracker. Designed for:
- short clips
- consistent-detection scenarios
- privacy-friendly appearance-based retrieval, not identity tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrackBox:
    """A single detection inside one frame, attached to a track."""

    track_id: int
    frame_idx: int
    timestamp_s: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float = 1.0
    label: str = ""


@dataclass
class TrackState:
    """Internal per-track state for the IoU tracker."""

    track_id: int
    last_box: tuple[float, float, float, float]
    last_frame_idx: int
    history: list[TrackBox] = field(default_factory=list)
    lost_frames: int = 0
    label: str = ""


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Standard axis-aligned IoU."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = a_area + b_area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


class SimpleIoUTracker:
    """Minimal multi-object tracker keyed on IoU.

    For each frame:
    - Build the cost matrix between active tracks' last boxes and new detections.
    - Greedy-match pairs above ``iou_threshold``.
    - Unmatched detections spawn new tracks; unmatched tracks accumulate lost
      frames and are pruned after ``max_lost_frames``.

    The output is a flat list of ``TrackBox`` covering every accepted detection,
    each carrying the assigned track_id.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 5,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
        if max_lost_frames < 0:
            raise ValueError(f"max_lost_frames must be >= 0, got {max_lost_frames}")
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self._tracks: dict[int, TrackState] = {}
        self._next_track_id = 1

    def update(
        self,
        detections: list[tuple[tuple[float, float, float, float], float, str]],
        *,
        frame_idx: int,
        timestamp_s: float,
    ) -> list[TrackBox]:
        """Update tracker with detections for one frame.

        ``detections`` is a list of ``(box, score, label)`` tuples.
        """
        if not self._tracks:
            return self._spawn_new(detections, frame_idx=frame_idx, timestamp_s=timestamp_s)

        active_ids = list(self._tracks.keys())

        # Greedy match: for each detection pick the best unused track > threshold.
        used_tracks: set[int] = set()
        assignments: list[tuple[int, int]] = []  # (detection_idx, track_id)
        for det_idx, (box, _score, _label) in enumerate(detections):
            best_iou = self.iou_threshold
            best_tid = -1
            for tid in active_ids:
                if tid in used_tracks:
                    continue
                iou = _iou(box, self._tracks[tid].last_box)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid != -1:
                assignments.append((det_idx, best_tid))
                used_tracks.add(best_tid)

        results: list[TrackBox] = []
        matched_dets: set[int] = set()

        for det_idx, tid in assignments:
            box, score, label = detections[det_idx]
            tb = TrackBox(
                track_id=tid,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                box=box,
                score=score,
                label=label,
            )
            state = self._tracks[tid]
            state.last_box = box
            state.last_frame_idx = frame_idx
            state.lost_frames = 0
            if label and not state.label:
                state.label = label
            state.history.append(tb)
            results.append(tb)
            matched_dets.add(det_idx)

        # Unmatched detections: spawn new tracks.
        unmatched = [d for i, d in enumerate(detections) if i not in matched_dets]
        results.extend(self._spawn_new(unmatched, frame_idx=frame_idx, timestamp_s=timestamp_s))

        # Bump lost counters and prune dead tracks.
        for tid in active_ids:
            if tid not in used_tracks:
                self._tracks[tid].lost_frames += 1
                if self._tracks[tid].lost_frames > self.max_lost_frames:
                    del self._tracks[tid]

        return results

    def _spawn_new(
        self,
        detections: list[tuple[tuple[float, float, float, float], float, str]],
        *,
        frame_idx: int,
        timestamp_s: float,
    ) -> list[TrackBox]:
        spawned: list[TrackBox] = []
        for box, score, label in detections:
            tid = self._next_track_id
            self._next_track_id += 1
            tb = TrackBox(
                track_id=tid,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                box=box,
                score=score,
                label=label,
            )
            self._tracks[tid] = TrackState(
                track_id=tid,
                last_box=box,
                last_frame_idx=frame_idx,
                history=[tb],
                label=label,
            )
            spawned.append(tb)
        return spawned

    def reset(self) -> None:
        """Clear all internal state. Useful between independent videos."""
        self._tracks.clear()
        self._next_track_id = 1


__all__ = ["SimpleIoUTracker", "TrackBox", "TrackState"]
