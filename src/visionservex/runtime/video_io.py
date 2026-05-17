# SPDX-License-Identifier: Apache-2.0
"""Unified video input abstraction.

Sources supported:
- webcam integer (``--source 0``, ``--source 1``)
- local video path (``input.mp4``, ``input.avi``, ``input.mkv``, ``input.mov``)
- RTSP / HTTP / MJPEG URL
- image folder or glob pattern (``frames/`` or ``frames/*.jpg``)

The OpenCV backend (cv2.VideoCapture) is loaded lazily so the base package
import is cheap. If OpenCV is missing, ``open_video_source`` raises a
structured ``OpenCVRequiredError`` that callers can catch.
"""

from __future__ import annotations

import contextlib
import glob
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class OpenCVRequiredError(ImportError):
    """cv2 is not installed."""

    code = "OPENCV_REQUIRED"
    fix = "pip install opencv-python-headless"


class VideoSourceOpenFailedError(Exception):
    """The source could not be opened."""

    def __init__(self, source: str, reason: str = "") -> None:
        super().__init__(f"VIDEO_SOURCE_OPEN_FAILED: {source}{(' — ' + reason) if reason else ''}")
        self.source = source
        self.reason = reason
        self.code = "VIDEO_SOURCE_OPEN_FAILED"


@dataclass
class VideoFrame:
    frame_index: int
    time_sec: float
    image: Any  # PIL.Image
    width: int
    height: int
    source: str


@dataclass
class VideoSource:
    """Iterator over frames from a unified source."""

    source: str
    max_frames: int | None = None
    start_sec: float = 0.0
    end_sec: float | None = None
    frame_stride: int = 1
    target_fps: float | None = None
    resize_width: int | None = None
    resize_height: int | None = None
    _capture: Any = None
    _frames_paths: list[str] | None = None
    _is_folder: bool = False
    _fps: float = 30.0

    def __iter__(self) -> Iterator[VideoFrame]:
        return self.iter_frames()

    def iter_frames(self) -> Iterator[VideoFrame]:
        if self._is_folder:
            yield from self._iter_folder()
        else:
            yield from self._iter_capture()

    def _resize_pil(self, img: Any) -> Any:
        if self.resize_width and self.resize_height:
            return img.resize((self.resize_width, self.resize_height))
        return img

    def _iter_folder(self) -> Iterator[VideoFrame]:
        from PIL import Image

        paths = self._frames_paths or []
        for i, p in enumerate(paths):
            if self.max_frames is not None and i >= self.max_frames:
                break
            if self.frame_stride > 1 and i % self.frame_stride != 0:
                continue
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            img = self._resize_pil(img)
            yield VideoFrame(
                frame_index=i,
                time_sec=i / max(self.target_fps or 1.0, 1.0),
                image=img,
                width=img.width,
                height=img.height,
                source=str(p),
            )

    def _iter_capture(self) -> Iterator[VideoFrame]:
        import cv2  # type: ignore
        import numpy as np  # noqa: F401
        from PIL import Image

        cap = self._capture
        if cap is None:
            return
        fps = self._fps if self._fps > 0 else 30.0
        if self.start_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start_sec * 1000.0)

        frame_index = 0
        emitted = 0
        last_emit_ts: float | None = None
        emit_interval = (1.0 / self.target_fps) if self.target_fps and self.target_fps > 0 else None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if self.max_frames is not None and emitted >= self.max_frames:
                break

            now_sec = frame_index / fps

            if self.end_sec is not None and now_sec > self.end_sec:
                break

            # frame stride
            if self.frame_stride > 1 and frame_index % self.frame_stride != 0:
                frame_index += 1
                continue

            # target_fps throttle (sample frames sparsely)
            if (
                emit_interval is not None
                and last_emit_ts is not None
                and now_sec - last_emit_ts < emit_interval
            ):
                frame_index += 1
                continue
            last_emit_ts = now_sec

            # BGR → RGB → PIL
            rgb = frame[:, :, ::-1]
            img = Image.fromarray(rgb)
            img = self._resize_pil(img)
            yield VideoFrame(
                frame_index=frame_index,
                time_sec=now_sec,
                image=img,
                width=img.width,
                height=img.height,
                source=self.source,
            )
            frame_index += 1
            emitted += 1

    def release(self) -> None:
        cap = self._capture
        self._capture = None
        if cap is not None:
            with contextlib.suppress(Exception):
                cap.release()


def open_video_source(
    source: str,
    *,
    max_frames: int | None = None,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    frame_stride: int = 1,
    target_fps: float | None = None,
    resize_width: int | None = None,
    resize_height: int | None = None,
    dry_run: bool = False,
) -> VideoSource:
    """Open a video source.

    Returns a VideoSource that can be iterated. If ``dry_run=True``, validates
    the source identifier without opening hardware (used for tests/CI).
    """
    s = VideoSource(
        source=source,
        max_frames=max_frames,
        start_sec=start_sec,
        end_sec=end_sec,
        frame_stride=frame_stride,
        target_fps=target_fps,
        resize_width=resize_width,
        resize_height=resize_height,
    )

    src = str(source).strip()

    # Folder / glob source
    if "*" in src or Path(src).is_dir():
        if "*" in src:
            paths = sorted(glob.glob(src))
        else:
            paths = sorted(
                p
                for p in (str(x) for x in Path(src).iterdir())
                if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            )
        if not paths and not dry_run:
            raise VideoSourceOpenFailedError(src, "no images found in folder/glob")
        s._frames_paths = paths
        s._is_folder = True
        return s

    if dry_run:
        # Validate the source kind without opening hardware.
        return s

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise OpenCVRequiredError("cv2 required for video / webcam / RTSP sources") from exc

    # Webcam integer
    capture_arg: int | str = src
    with contextlib.suppress(ValueError):
        capture_arg = int(src)

    cap = cv2.VideoCapture(capture_arg)
    if not cap.isOpened():
        raise VideoSourceOpenFailedError(src, "cv2.VideoCapture could not open")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    s._capture = cap
    s._fps = float(fps if fps > 0 else 30.0)
    return s


def make_synthetic_video(
    out_path: str | Path,
    *,
    frames: int = 30,
    width: int = 640,
    height: int = 360,
    objects: int = 2,
    fps: float = 15.0,
) -> Path:
    """Create a tiny synthetic .mp4 video for tests/CI.

    Falls back to writing an image sequence under a folder if cv2 cannot
    encode mp4 (e.g., codec missing on the runner).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore
        import numpy as np
    except ImportError as exc:
        raise OpenCVRequiredError("cv2 required for synthetic video generation") from exc

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
    if not writer.isOpened():
        # Fallback: image sequence
        seq_dir = out.with_suffix("")
        seq_dir.mkdir(exist_ok=True)
        for i in range(frames):
            img = np.full((height, width, 3), 32, dtype=np.uint8)
            for k in range(objects):
                x = int((i * 7 + k * 80) % (width - 80))
                y = int((50 + k * 60) % (height - 60))
                img[y : y + 60, x : x + 80] = (60 + k * 40, 200 - k * 40, 100)
            cv2.imwrite(str(seq_dir / f"frame_{i:04d}.png"), img)
        return seq_dir

    for i in range(frames):
        import numpy as np

        img = np.full((height, width, 3), 32, dtype=np.uint8)
        for k in range(objects):
            x = int((i * 7 + k * 80) % (width - 80))
            y = int((50 + k * 60) % (height - 60))
            img[y : y + 60, x : x + 80] = (60 + k * 40, 200 - k * 40, 100)
        writer.write(img)
    writer.release()
    return out


__all__ = [
    "OpenCVRequiredError",
    "VideoFrame",
    "VideoSource",
    "VideoSourceOpenFailedError",
    "make_synthetic_video",
    "open_video_source",
]
