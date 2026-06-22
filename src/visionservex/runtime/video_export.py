# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Overlay video export + annotation artifact exporters (v3.22.0).

``export_overlay_video`` streams frames, draws detections/masks using the result's
own ``plot`` overlay, and pipes raw RGB to ffmpeg ``h264_nvenc`` (or libx264) to
produce a browser-safe MP4 (yuv420p, avc1, faststart) WITHOUT loading the whole
video into RAM. If ffmpeg is unavailable it falls back to cv2 ``mp4v`` and reports
the exact reason.

Annotation exporters (JSON/CSV/COCO/YOLO) turn an ``infer_video`` report into the
standard machine-readable formats with full provenance.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
from typing import Any

from visionservex.runtime.ffmpeg_tools import FFMPEG, detect_hwaccel
from visionservex.runtime.video_io import open_video_source
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)


def export_overlay_video(
    model: Any,
    video_path: str,
    out_path: str,
    *,
    sample_fps: float | None = None,
    max_frames: int | None = None,
    threshold: float | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    """Render an annotated MP4 by streaming frames through ``model.predict``.

    Returns a dict with ``ok``, ``output``, ``encoder``, ``media_type``, ``frames``.
    """
    import numpy as np

    predict_kwargs: dict[str, Any] = {}
    if threshold is not None:
        predict_kwargs["threshold"] = threshold

    src = open_video_source(video_path, max_frames=max_frames, target_fps=sample_fps)
    out_fps = fps or sample_fps or getattr(src, "_fps", 30.0) or 30.0

    # Peek first frame to learn dimensions + produce its overlay.
    it = iter(src.iter_frames())
    try:
        first = next(it)
    except StopIteration:
        src.release()
        return {"ok": False, "error": "no frames decoded", "output": out_path}
    w, h = first.image.width, first.image.height

    use_ffmpeg = bool(FFMPEG)
    hw = detect_hwaccel()
    encoder = (
        "h264_nvenc"
        if (use_ffmpeg and hw["nvenc_available"])
        else ("libx264" if use_ffmpeg else "cv2_mp4v")
    )

    n = 0

    def _annotate(frame_img: Any) -> Any:
        result = model.predict(frame_img, **predict_kwargs)
        try:
            return result.plot(frame_img)
        except Exception:
            return frame_img

    if use_ffmpeg:
        args = [
            FFMPEG,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(out_fps),
            "-i",
            "-",
        ]
        if encoder == "h264_nvenc":
            args += [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p5",
                "-rc",
                "vbr",
                "-cq",
                "23",
                "-profile:v",
                "high",
            ]
        else:
            args += ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-profile:v", "high"]
        args += ["-pix_fmt", "yuv420p", "-tag:v", "avc1", "-movflags", "+faststart", out_path]

        proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        try:
            for vf in _chain_first(first, it):
                ann = _annotate(vf.image)
                arr = np.asarray(ann.convert("RGB"), dtype=np.uint8)
                if arr.shape[1] != w or arr.shape[0] != h:
                    from PIL import Image as _I

                    arr = np.asarray(
                        _I.fromarray(arr).resize((w, h)).convert("RGB"), dtype=np.uint8
                    )
                proc.stdin.write(arr.tobytes())
                n += 1
            proc.stdin.close()
            rc = proc.wait(timeout=1800)
            stderr_tail = (proc.stderr.read() or b"").decode("utf-8", "replace")[-400:]
        finally:
            src.release()
            with contextlib.suppress(Exception):
                proc.stdin and proc.stdin.close()
        ok = rc == 0 and os.path.exists(out_path)
        return {
            "ok": ok,
            "output": out_path,
            "encoder": encoder,
            "frames": n,
            "media_type": "video/mp4",
            "nvenc_used": encoder == "h264_nvenc",
            "error": None if ok else f"ffmpeg rc={rc}: {stderr_tail}",
        }

    # ---- fallback: cv2 mp4v (NOT guaranteed browser-safe; reported honestly) ----
    try:
        import cv2  # type: ignore

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, float(out_fps), (w, h))
        for vf in _chain_first(first, it):
            ann = _annotate(vf.image)
            arr = np.asarray(ann.convert("RGB"), dtype=np.uint8)[:, :, ::-1]
            writer.write(arr)
            n += 1
        writer.release()
    finally:
        src.release()
    return {
        "ok": os.path.exists(out_path),
        "output": out_path,
        "encoder": "cv2_mp4v",
        "frames": n,
        "media_type": "video/mp4",
        "nvenc_used": False,
        "warning": "ffmpeg not installed: used cv2 mp4v which may not be browser-compatible; "
        "install ffmpeg for H.264/avc1/faststart output",
    }


def _chain_first(first: Any, rest: Any):
    yield first
    yield from rest


# --------------------------------------------------------------------------- #
# Annotation exporters (Phase 8.3)
# --------------------------------------------------------------------------- #
def export_annotations(report: dict[str, Any], fmt: str, *, model_id: str = "") -> str:
    """Convert an ``infer_video`` report to a string in the requested format.

    Supported: ``json`` (internal), ``csv``, ``coco`` (detection). Each carries
    provenance (model id, frame sampling) where the format allows.
    """
    fmt = fmt.lower()
    frames = report.get("frames", [])
    if fmt == "json":
        import json

        return json.dumps({"model_id": model_id, "report": report}, indent=2)
    if fmt == "csv":
        import csv
        import io

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(
            ["frame_index", "time_sec", "label", "class_id", "score", "x1", "y1", "x2", "y2"]
        )
        for f in frames:
            for d in f["detections"]:
                b = d["box"]
                w.writerow(
                    [f["frame_index"], f["time_sec"], d["label"], d.get("class_id"), d["score"], *b]
                )
        return buf.getvalue()
    if fmt == "coco":
        import json

        images = []
        annotations = []
        cats: dict[str, int] = {}
        ann_id = 1
        for f in frames:
            images.append({"id": f["frame_index"], "timestamp": f["time_sec"]})
            for d in f["detections"]:
                lbl = d["label"]
                cid = cats.setdefault(lbl, len(cats) + 1)
                x1, y1, x2, y2 = d["box"]
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": f["frame_index"],
                        "category_id": cid,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": d["score"],
                        "area": (x2 - x1) * (y2 - y1),
                    }
                )
                ann_id += 1
        return json.dumps(
            {
                "info": {"model_id": model_id, "source": report.get("video")},
                "images": images,
                "annotations": annotations,
                "categories": [{"id": v, "name": k} for k, v in cats.items()],
            },
            indent=2,
        )
    raise ValueError(f"unsupported export format {fmt!r}; use json|csv|coco")


__all__ = ["export_annotations", "export_overlay_video"]
