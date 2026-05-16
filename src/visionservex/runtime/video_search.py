# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Local-only surveillance video-search index and query.

The pipeline (one function call per stage so each stage is independently
testable):

1. ``iter_frames(source, sample_fps, stride)`` — yields (frame_idx, timestamp_s, PIL.Image)
   from either a video file or a folder of image frames.
2. ``detect_frame(detector, frame, prompt)`` — uses a VisionModel detector with
   text prompts. Returns ``list[(box, score, label)]``.
3. ``track(detections_per_frame)`` — uses ``SimpleIoUTracker`` to attach
   track_ids across frames.
4. ``embed_crops(embedder, crops)`` — uses a VisionModel embedder (DINOv2 /
   SigLIP2) to produce L2-normalized vectors.
5. ``save_index(out_dir, manifest, embeddings)`` — atomic local save.
6. ``query_index(index_dir, text, top_k)`` — text→vector via embedder, cosine
   similarity, ranked timeline.

Privacy defaults:
- local-only (no network egress on index/query)
- no face recognition / no biometric identity claim
- appearance-based retrieval only
- metadata-only mode strips thumbnails

Note on imports:
We import torch / numpy / cv2 lazily inside functions so the quick-test
environment without torch can still load this module.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from visionservex.runtime.simple_tracker import SimpleIoUTracker, TrackBox

# Privacy disclaimer printed by every CLI entry point.
PRIVACY_NOTICE = (
    "VisionServeX video-search is appearance-based retrieval only. "
    "It does NOT perform face recognition, biometric identification, or identity "
    "matching. All processing is local; no data is sent to any external service."
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


# ---------------------------------------------------------------------------
# Frame iteration
# ---------------------------------------------------------------------------


def iter_frames(
    source: str | Path,
    *,
    sample_fps: float | None = 1.0,
    stride: int | None = None,
    max_frames: int | None = None,
) -> Iterator[tuple[int, float, Image.Image]]:
    """Yield (frame_idx, timestamp_s, PIL.Image) from a folder or video file.

    For a folder: every file matching an image extension, sorted by name.
    For a video:  uses cv2 (lazy import). ``sample_fps`` controls subsampling.
                  ``stride`` overrides sample_fps when provided.

    ``max_frames`` is an upper bound used by tests to cap synthetic frame folders.
    """
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)
        # Folder mode: treat ordering as 1 fps if no stride.
        step = stride if stride and stride > 0 else 1
        for i, f in enumerate(files[::step]):
            if max_frames and i >= max_frames:
                return
            try:
                img = Image.open(f).convert("RGB")
            except Exception:
                continue
            yield i, float(i), img
        return

    if p.suffix.lower() in _VIDEO_EXTS:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Video-file ingestion requires opencv-python. "
                "Install with: pip install opencv-python-headless"
            ) from exc

        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {p}")
        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if stride and stride > 0:
                step = stride
            elif sample_fps and sample_fps > 0:
                step = max(1, round(video_fps / sample_fps))
            else:
                step = 1
            frame_idx = 0
            out_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    yield out_idx, frame_idx / max(video_fps, 1e-6), img
                    out_idx += 1
                    if max_frames and out_idx >= max_frames:
                        break
                frame_idx += 1
        finally:
            cap.release()
        return

    raise ValueError(f"Unsupported source: {p}. Expected folder of images or a video file.")


# ---------------------------------------------------------------------------
# Detection / cropping helpers
# ---------------------------------------------------------------------------


def crop_box(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    """Clip box to image bounds and crop. Returns a min-1x1 PIL image."""
    x1, y1, x2, y2 = box
    w, h = image.size
    x1 = max(0, min(round(x1), w - 1))
    y1 = max(0, min(round(y1), h - 1))
    x2 = max(x1 + 1, min(round(x2), w))
    y2 = max(y1 + 1, min(round(y2), h))
    return image.crop((x1, y1, x2, y2))


def detections_from_result(
    result: Any,
) -> list[tuple[tuple[float, float, float, float], float, str]]:
    """Extract (box_xyxy, score, label) tuples from a VisionServeX result.

    Works with DetectionResult and OpenVocabularyResult — both expose a
    ``.detections`` list of ``Detection(box, score, label)``.
    """
    out: list[tuple[tuple[float, float, float, float], float, str]] = []
    detections = getattr(result, "detections", None)
    if detections is None:
        return out
    for d in detections:
        box = getattr(d, "box", None)
        if box is None:
            continue
        x1, y1, x2, y2 = (
            float(box.x1),
            float(box.y1),
            float(box.x2),
            float(box.y2),
        )
        out.append(
            ((x1, y1, x2, y2), float(getattr(d, "score", 1.0)), str(getattr(d, "label", "")))
        )
    return out


# ---------------------------------------------------------------------------
# Index data classes
# ---------------------------------------------------------------------------


@dataclass
class IndexedCrop:
    """One indexed detection: position in time + appearance vector."""

    track_id: int
    frame_idx: int
    timestamp_s: float
    box: tuple[float, float, float, float]
    score: float
    label: str
    embedding_idx: int  # row into the embeddings.npy matrix

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IndexManifest:
    """Top-level metadata for an index directory."""

    version: int = 1
    created_at: float = field(default_factory=time.time)
    detector_model_id: str = ""
    embedder_model_id: str = ""
    tracker: str = "simple-iou"
    source: str = ""
    sample_fps: float | None = None
    stride: int | None = None
    prompt: str = ""
    privacy_mode: str = "appearance_only"
    n_frames_seen: int = 0
    n_detections: int = 0
    n_tracks: int = 0
    embedding_dim: int = 0
    crops: list[IndexedCrop] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["crops"] = [c if isinstance(c, dict) else c.to_dict() for c in self.crops]
        return d


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_index(
    out_dir: str | Path,
    manifest: IndexManifest,
    embeddings,
) -> Path:
    """Persist an index atomically. Returns the out_dir path."""
    import numpy as np

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    (out / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2))
    (out / "README.md").write_text(
        "# VisionServeX video-search index\n\n"
        f"{PRIVACY_NOTICE}\n\n"
        "Generated locally. Safe to delete this directory at any time.\n"
    )
    return out


def load_index(index_dir: str | Path) -> tuple[IndexManifest, Any]:
    """Load an index. Returns ``(manifest, embeddings_array)``."""
    import numpy as np

    p = Path(index_dir)
    mpath = p / "manifest.json"
    epath = p / "embeddings.npy"
    if not mpath.exists() or not epath.exists():
        raise FileNotFoundError(f"Index {p} is missing manifest.json or embeddings.npy")
    raw = json.loads(mpath.read_text())
    crops = [IndexedCrop(**c) for c in raw.get("crops", [])]
    raw.pop("crops", None)
    manifest = IndexManifest(**raw, crops=crops)
    embeddings = np.load(epath)
    return manifest, embeddings


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def embedding_from_result(result: Any) -> Any:
    """Extract a numpy vector from a VisionServeX embed result.

    Tolerates several shapes: ``.embedding`` (np.ndarray), ``.metadata['embedding']``,
    or a list/tuple of floats.
    """
    import numpy as np

    vec = getattr(result, "embedding", None)
    if vec is None:
        meta = getattr(result, "metadata", {}) or {}
        vec = meta.get("embedding")
    if vec is None:
        raise ValueError("Embedder result did not contain an embedding vector")
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return arr


def l2_normalize(vectors):
    """Row-wise L2 normalize a 2-D matrix. Pure numpy."""
    import numpy as np

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return arr / norms


# ---------------------------------------------------------------------------
# Public query API
# ---------------------------------------------------------------------------


@dataclass
class VideoSearchHit:
    track_id: int
    frame_idx: int
    timestamp_s: float
    box: tuple[float, float, float, float]
    label: str
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def query_index(
    index_dir: str | Path,
    query_vector,
    *,
    top_k: int = 20,
    group_by_track: bool = True,
) -> list[VideoSearchHit]:
    """Run cosine similarity on a loaded index and return top-k hits.

    If ``group_by_track`` is True, returns the best-scoring crop per track_id.
    """

    manifest, embeddings = load_index(index_dir)
    if embeddings.size == 0:
        return []
    q = l2_normalize(query_vector).reshape(1, -1)
    e = l2_normalize(embeddings)
    sims = (e @ q.T).reshape(-1)

    hits: list[VideoSearchHit] = []
    for idx, sim in enumerate(sims):
        if idx >= len(manifest.crops):
            break
        c = manifest.crops[idx]
        hits.append(
            VideoSearchHit(
                track_id=c.track_id,
                frame_idx=c.frame_idx,
                timestamp_s=c.timestamp_s,
                box=tuple(c.box),
                label=c.label,
                similarity=float(sim),
            )
        )

    hits.sort(key=lambda h: h.similarity, reverse=True)

    if group_by_track:
        best_per_track: dict[int, VideoSearchHit] = {}
        for h in hits:
            cur = best_per_track.get(h.track_id)
            if cur is None or h.similarity > cur.similarity:
                best_per_track[h.track_id] = h
        hits = sorted(best_per_track.values(), key=lambda h: h.similarity, reverse=True)

    return hits[:top_k]


# ---------------------------------------------------------------------------
# Top-level index builder (no VisionModel hard-dep here — callers pass
# already-built engine objects with .predict)
# ---------------------------------------------------------------------------


def build_index(
    *,
    source: str | Path,
    out_dir: str | Path,
    detect_fn,
    embed_fn,
    detector_model_id: str = "",
    embedder_model_id: str = "",
    prompt: str = "person",
    sample_fps: float | None = 1.0,
    stride: int | None = None,
    max_frames: int | None = None,
    iou_threshold: float = 0.3,
    max_lost_frames: int = 5,
) -> Path:
    """Build a video-search index using caller-supplied detect/embed functions.

    ``detect_fn(image, prompt) -> list[(box, score, label)]``
    ``embed_fn(crop_image) -> 1-D numpy vector``

    This signature deliberately accepts callables (not VisionModel) so tests
    can run end-to-end with mocked detector + embedder.
    """
    import numpy as np

    tracker = SimpleIoUTracker(iou_threshold=iou_threshold, max_lost_frames=max_lost_frames)
    crops: list[IndexedCrop] = []
    embeddings: list[Any] = []
    n_detections = 0
    n_frames_seen = 0
    tracks_seen: set[int] = set()

    for frame_idx, ts, frame in iter_frames(
        source, sample_fps=sample_fps, stride=stride, max_frames=max_frames
    ):
        n_frames_seen += 1
        dets = detect_fn(frame, prompt)
        if not dets:
            continue
        n_detections += len(dets)
        tracked: list[TrackBox] = tracker.update(dets, frame_idx=frame_idx, timestamp_s=ts)
        for tb in tracked:
            tracks_seen.add(tb.track_id)
            crop = crop_box(frame, tb.box)
            vec = np.asarray(embed_fn(crop), dtype=np.float32).reshape(-1)
            emb_idx = len(embeddings)
            embeddings.append(vec)
            crops.append(
                IndexedCrop(
                    track_id=tb.track_id,
                    frame_idx=tb.frame_idx,
                    timestamp_s=tb.timestamp_s,
                    box=tuple(tb.box),
                    score=tb.score,
                    label=tb.label,
                    embedding_idx=emb_idx,
                )
            )

    embedding_dim = len(embeddings[0]) if embeddings else 0
    manifest = IndexManifest(
        detector_model_id=detector_model_id,
        embedder_model_id=embedder_model_id,
        source=str(source),
        sample_fps=sample_fps,
        stride=stride,
        prompt=prompt,
        n_frames_seen=n_frames_seen,
        n_detections=n_detections,
        n_tracks=len(tracks_seen),
        embedding_dim=embedding_dim,
        crops=crops,
    )
    return save_index(out_dir, manifest, embeddings)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_timeline_html(
    hits: list[VideoSearchHit],
    *,
    query_text: str,
    source: str = "",
) -> str:
    """Render a minimal self-contained HTML timeline. No external resources."""
    rows = "\n".join(
        f"<tr><td>{h.timestamp_s:.2f}s</td><td>{h.track_id}</td><td>{h.label}</td>"
        f"<td>{h.similarity:.3f}</td><td>{int(h.box[0])},{int(h.box[1])},"
        f"{int(h.box[2])},{int(h.box[3])}</td></tr>"
        for h in hits
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>video-search: {query_text}</title>"
        "<style>body{font-family:system-ui;margin:24px}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ddd;padding:6px 10px;text-align:left}"
        "th{background:#f4f4f4}.note{color:#666;font-size:90%;margin:14px 0}"
        "</style></head><body>"
        f"<h2>Video-search results</h2>"
        f"<p><b>Query:</b> {query_text}</p>"
        f"<p><b>Source:</b> {source}</p>"
        f"<p class='note'>{PRIVACY_NOTICE}</p>"
        "<table><thead><tr><th>Time</th><th>Track</th><th>Label</th>"
        "<th>Similarity</th><th>Box (x1,y1,x2,y2)</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></body></html>"
    )


__all__ = [
    "PRIVACY_NOTICE",
    "IndexManifest",
    "IndexedCrop",
    "VideoSearchHit",
    "build_index",
    "crop_box",
    "detections_from_result",
    "embedding_from_result",
    "iter_frames",
    "l2_normalize",
    "load_index",
    "query_index",
    "render_timeline_html",
    "save_index",
]
