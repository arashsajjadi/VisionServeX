# SPDX-License-Identifier: Apache-2.0
"""Live inference runtime.

Loads a single VisionModel once and drives it frame-by-frame against a
``VideoSource``. Writes annotated frames to MP4 and per-frame JSONL when
configured. Supports adaptive FPS and structured failures.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LiveConfig:
    source: str
    model_id: str
    task: str = "detect"
    prompt: str | None = None
    device: str = "auto"
    confidence: float = 0.25
    tracker: str | None = None
    max_frames: int | None = None
    target_fps: float | None = None
    frame_stride: int = 1
    start_sec: float = 0.0
    end_sec: float | None = None
    resize_width: int | None = None
    resize_height: int | None = None
    display: bool = False
    headless: bool = True
    out_video: str | None = None
    json_out: str | None = None
    save_frames: bool = False
    frame_output_dir: str | None = None
    show_fps: bool = False
    skip_frames_when_slow: bool = False
    warmup_frames: int = 0
    dry_run: bool = False


@dataclass
class LiveResult:
    source: str
    model_id: str
    task: str
    frames_read: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    average_fps: float = 0.0
    median_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0
    output_video: str | None = None
    json_out: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _per_frame_payload(
    result: Any, *, frame_index: int, time_sec: float, runtime_ms: float, config: LiveConfig
) -> dict[str, Any]:
    """Convert a VisionModel result into the canonical per-frame envelope."""
    payload: dict[str, Any] = {
        "frame_index": frame_index,
        "time_sec": round(time_sec, 4),
        "runtime_ms": round(runtime_ms, 2),
        "model_id": config.model_id,
        "task": config.task,
        "detections": [],
        "masks": [],
        "tracks": [],
        "pose": [],
        "oriented_boxes": [],
        "warnings": [],
        "errors": [],
    }
    if result is None:
        payload["warnings"].append("no_predictions")
        return payload

    # Detection-style results
    if hasattr(result, "detections"):
        for d in result.detections or []:
            box = getattr(d, "box", None)
            if box is None:
                continue
            payload["detections"].append(
                {
                    "box": {
                        "x1": float(getattr(box, "x1", 0)),
                        "y1": float(getattr(box, "y1", 0)),
                        "x2": float(getattr(box, "x2", 0)),
                        "y2": float(getattr(box, "y2", 0)),
                    },
                    "score": float(getattr(d, "score", 0.0)),
                    "class_id": getattr(d, "class_id", None),
                    "class_name": getattr(d, "label", None) or getattr(d, "phrase", None) or "",
                }
            )
    # Segmentation results
    if hasattr(result, "segments"):
        import numpy as np

        for i, seg in enumerate(result.segments or []):
            mask = getattr(seg, "mask", None)
            box_obj = getattr(seg, "box", None)
            payload["masks"].append(
                {
                    "mask_shape": list(np.asarray(mask).shape) if mask is not None else None,
                    "box": (
                        {
                            "x1": float(getattr(box_obj, "x1", 0)),
                            "y1": float(getattr(box_obj, "y1", 0)),
                            "x2": float(getattr(box_obj, "x2", 0)),
                            "y2": float(getattr(box_obj, "y2", 0)),
                        }
                        if box_obj is not None
                        else None
                    ),
                    "score": float(getattr(seg, "score", 0.0)),
                    "class_name": getattr(seg, "label", None) or "",
                    "index": i,
                }
            )
    return payload


def run_live(config: LiveConfig) -> Iterator[dict[str, Any]]:
    """Yield per-frame payloads.

    The caller is responsible for writing them to JSONL/video. ``run_live`` is
    a pure generator so it can be wrapped in adaptive throttling or display
    loops without coupling to the CLI.
    """
    from visionservex.runtime.video_io import (
        OpenCVRequiredError,
        VideoSourceOpenFailedError,
        open_video_source,
    )

    if config.dry_run:
        # Validate config / source without loading the model.
        try:
            open_video_source(config.source, dry_run=True)
        except VideoSourceOpenFailedError as exc:
            yield {"errors": [exc.code], "source": config.source}
            return
        yield {
            "code": "DRY_RUN_OK",
            "source": config.source,
            "model_id": config.model_id,
            "task": config.task,
        }
        return

    # Load model once
    try:
        from visionservex import VisionModel

        model = VisionModel(config.model_id)
    except Exception as exc:
        yield {"errors": ["MODEL_LOAD_FAILED"], "message": str(exc)[:200]}
        return

    try:
        source = open_video_source(
            config.source,
            max_frames=config.max_frames,
            start_sec=config.start_sec,
            end_sec=config.end_sec,
            frame_stride=config.frame_stride,
            target_fps=config.target_fps,
            resize_width=config.resize_width,
            resize_height=config.resize_height,
        )
    except OpenCVRequiredError as exc:
        yield {"errors": [exc.code], "fix": exc.fix}
        return
    except VideoSourceOpenFailedError as exc:
        yield {"errors": [exc.code], "source": config.source, "reason": exc.reason}
        return

    inference_ms_list: list[float] = []
    try:
        for frame in source:
            t0 = time.time()
            try:
                kwargs: dict[str, Any] = {}
                if config.task in ("open-vocab", "open_vocab", "open_vocab_detect") and config.prompt:
                    kwargs["prompt"] = config.prompt
                result = model.predict(frame.image, **kwargs)
                runtime_ms = (time.time() - t0) * 1000.0
                inference_ms_list.append(runtime_ms)
                payload = _per_frame_payload(
                    result, frame_index=frame.frame_index, time_sec=frame.time_sec,
                    runtime_ms=runtime_ms, config=config,
                )
            except Exception as exc:  # pragma: no cover - per-frame error
                payload = {
                    "frame_index": frame.frame_index,
                    "time_sec": frame.time_sec,
                    "runtime_ms": (time.time() - t0) * 1000.0,
                    "errors": ["FRAME_INFERENCE_FAILED"],
                    "message": str(exc)[:200],
                    "model_id": config.model_id,
                    "task": config.task,
                    "detections": [],
                    "masks": [],
                    "tracks": [],
                    "pose": [],
                    "oriented_boxes": [],
                    "warnings": [],
                }
            yield payload
    finally:
        source.release()


def summarize_live(payloads: list[dict[str, Any]], config: LiveConfig) -> LiveResult:
    """Aggregate per-frame payloads into a summary result."""
    inf = [p.get("runtime_ms", 0.0) for p in payloads if "runtime_ms" in p]
    res = LiveResult(
        source=config.source,
        model_id=config.model_id,
        task=config.task,
        frames_read=len(payloads),
        frames_processed=sum(1 for p in payloads if not p.get("errors")),
        frames_dropped=sum(1 for p in payloads if p.get("errors")),
        output_video=config.out_video,
        json_out=config.json_out,
    )
    if inf:
        total_sec = sum(inf) / 1000.0
        res.average_fps = round(len(inf) / total_sec, 2) if total_sec > 0 else 0.0
        res.median_inference_ms = round(statistics.median(inf), 2)
        if len(inf) >= 20:
            res.p95_inference_ms = round(statistics.quantiles(inf, n=20)[-1], 2)
        else:
            res.p95_inference_ms = round(max(inf), 2)
    return res


__all__ = ["LiveConfig", "LiveResult", "run_live", "summarize_live"]
