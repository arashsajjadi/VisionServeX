# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Worker-side video inference pipeline (v3.22.0).

Streams frames from disk (never loads the whole video into RAM), groups them into
adaptive microbatches sized by :class:`AdaptiveBatchScheduler`, runs each wave
through the model's ``batch_predict`` (TRUE tensor batch for D-FINE), and yields
per-frame results that are independent and carry their original frame index +
timestamp.

Ties together:
* Phase 2 — ``run_batch_with_telemetry`` (true batch + measured metadata)
* Phase 3 — ``AdaptiveBatchScheduler`` (safe ladder, bottleneck attribution)
* Phase 7 — :class:`CancelToken` (abort between waves, release tensors)

Each wave releases its temporary tensors and (periodically) the CUDA cache, so a
cancel or completion returns VRAM promptly.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from visionservex.runtime.adaptive_batch import (
    AdaptiveBatchScheduler,
    SchedulerConfig,
    WaveOutcome,
)
from visionservex.runtime.batch_infer import run_batch_with_telemetry
from visionservex.runtime.gpu_lifecycle import (
    clear_torch_cuda_cache,
    force_gc,
    get_gpu_memory_state,
)


@dataclass
class CancelToken:
    """Cooperative cancellation flag checked between waves."""

    _event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass
class FrameResult:
    frame_index: int
    time_sec: float
    detections: list[dict[str, Any]] = field(default_factory=list)
    n_objects: int = 0
    raw: Any = None  # the BaseResult (optional; dropped for big runs)


def _stream_frames(
    video_path: str,
    *,
    sample_fps: float | None,
    stride: int,
    start_s: float,
    end_s: float | None,
    max_frames: int | None,
) -> Iterator[tuple[int, float, Any]]:
    """Yield (frame_index, time_sec, PIL.Image) one at a time (streaming)."""
    from visionservex.runtime.video_io import open_video_source

    src = open_video_source(
        video_path,
        max_frames=max_frames,
        start_sec=start_s,
        end_sec=end_s,
        frame_stride=stride,
        target_fps=sample_fps,
    )
    try:
        for vf in src.iter_frames():
            yield vf.frame_index, vf.time_sec, vf.image
    finally:
        src.release()


def _result_to_frame(result: Any, frame_index: int, time_sec: float, keep_raw: bool) -> FrameResult:
    """Project a BaseResult into a lightweight per-frame record (drops heavy refs)."""
    dets: list[dict[str, Any]] = []
    items = getattr(result, "detections", None)
    if items is not None:
        for d in items:
            b = d.box
            dets.append(
                {
                    "box": [b.x1, b.y1, b.x2, b.y2],
                    "score": float(d.score),
                    "label": d.label,
                    "class_id": d.class_id,
                }
            )
    else:
        # segmentation result → boxes + (mask handled by Phase 5 serialization)
        for s in getattr(result, "segments", []) or []:
            b = s.box
            dets.append(
                {
                    "box": [b.x1, b.y1, b.x2, b.y2],
                    "score": float(s.score),
                    "label": s.label,
                    "class_id": s.class_id,
                    "has_mask": getattr(s, "mask", None) is not None
                    and getattr(s.mask, "size", 0) > 1,
                }
            )
    # drop the held PIL image to avoid retaining decoded frames in RAM
    with contextlib.suppress(Exception):
        result._image = None
    return FrameResult(
        frame_index=frame_index,
        time_sec=round(time_sec, 4),
        detections=dets,
        n_objects=len(dets),
        raw=result if keep_raw else None,
    )


def infer_video(
    model: Any,
    video_path: str,
    *,
    sample_fps: float | None = None,
    stride: int = 1,
    start_s: float = 0.0,
    end_s: float | None = None,
    max_frames: int | None = None,
    mode: str = "balanced",
    threshold: float | None = None,
    cancel: CancelToken | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    vram_total_mb: float | None = None,
    keep_raw: bool = False,
    cache_release_every: int = 8,
    **predict_kwargs: Any,
) -> dict[str, Any]:
    """Run adaptive-batched inference over a whole video, streaming from disk.

    Returns a dict with per-frame results, the batch-size trajectory, scheduler
    decisions, aggregate telemetry, and whether the run was cancelled. Frame
    results are INDEPENDENT — each frame's detections derive only from its own
    pixels at the per-frame ``threshold``.
    """
    if vram_total_mb is None:
        mem = get_gpu_memory_state()
        # nvml total if available, else a conservative default
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            vram_total_mb = pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024**2)
        except Exception:
            vram_total_mb = 16000.0 if mem.cuda_available else 1.0

    max_b = int(getattr(model, "max_batch_size_hint", 8)) or 8
    sched = AdaptiveBatchScheduler(
        SchedulerConfig(vram_total_mb=float(vram_total_mb), mode=mode, model_max_batch=max_b)  # type: ignore[arg-type]
    )

    if threshold is not None:
        predict_kwargs.setdefault("threshold", threshold)

    frames_out: list[FrameResult] = []
    decisions: list[dict[str, Any]] = []
    trajectory: list[int] = []
    waves = 0
    total_forward_ms = 0.0
    total_preprocess_ms = 0.0
    total_postprocess_ms = 0.0
    util_samples: list[float] = []
    vram_peak = 0.0
    cancelled = False
    t_start = time.perf_counter()

    buf: list[tuple[int, float, Any]] = []

    def _flush() -> bool:
        """Run one wave on the buffer. Returns False if cancelled."""
        nonlocal waves, total_forward_ms, total_preprocess_ms, total_postprocess_ms, vram_peak
        if not buf:
            return True
        if cancel is not None and cancel.cancelled:
            return False
        idxs = [b[0] for b in buf]
        tss = [b[1] for b in buf]
        imgs = [b[2] for b in buf]
        results, tel = run_batch_with_telemetry(model, imgs, **predict_kwargs)
        bs = len(imgs)
        waves += 1
        trajectory.append(bs)
        total_forward_ms += tel.forward_ms
        total_preprocess_ms += tel.preprocess_ms
        total_postprocess_ms += tel.postprocess_ms
        if tel.gpu_util_avg is not None:
            util_samples.append(tel.gpu_util_avg)
        vram_peak = max(vram_peak, tel.vram_used_peak_mb)
        for fi, ts, res in zip(idxs, tss, results, strict=False):
            frames_out.append(_result_to_frame(res, fi, ts, keep_raw))
        # feed scheduler
        fps = bs / (tel.total_ms / 1000.0) if tel.total_ms > 0 else 0.0
        outcome = WaveOutcome(
            batch_size=bs,
            throughput_fps=fps,
            latency_ms=tel.total_ms,
            vram_used_mb=tel.vram_used_peak_mb,
            vram_free_mb=tel.vram_free_min_mb or 0.0,
            gpu_util_avg=tel.gpu_util_avg,
            preprocess_ms=tel.preprocess_ms,
            forward_ms=tel.forward_ms,
            postprocess_ms=tel.postprocess_ms,
            cancel_requested=bool(cancel is not None and cancel.cancelled),
        )
        d = sched.record(outcome)
        decisions.append(
            {
                "wave": waves,
                "batch_size": bs,
                "throughput_fps": round(fps, 1),
                "action": d.action,
                "bottleneck": d.bottleneck,
                "reason": d.reason,
                "next_batch_size": d.next_batch_size,
            }
        )
        if on_progress is not None:
            on_progress(
                {
                    "frames_done": len(frames_out),
                    "waves": waves,
                    "batch_size": bs,
                    "throughput_fps": round(fps, 1),
                    "bottleneck": d.bottleneck,
                    "next_batch_size": d.next_batch_size,
                }
            )
        buf.clear()
        # release temporary tensors each wave; CUDA cache periodically
        force_gc()
        if waves % max(1, cache_release_every) == 0:
            clear_torch_cuda_cache()
        return not (cancel is not None and cancel.cancelled)

    for fi, ts, img in _stream_frames(
        video_path,
        sample_fps=sample_fps,
        stride=stride,
        start_s=start_s,
        end_s=end_s,
        max_frames=max_frames,
    ):
        if cancel is not None and cancel.cancelled:
            cancelled = True
            break
        buf.append((fi, ts, img))
        if len(buf) >= sched.batch_size and not _flush():
            cancelled = True
            break
    else:
        # drain remaining frames (only if not cancelled)
        if not (cancel is not None and cancel.cancelled):
            _flush()
    if cancelled:
        buf.clear()

    # final memory release (Phase 3.6 / Phase 7 cleanup)
    force_gc()
    clear_torch_cuda_cache()
    after = get_gpu_memory_state("video_after_cleanup")

    elapsed = time.perf_counter() - t_start
    n = len(frames_out)
    return {
        "video": video_path,
        "frames_processed": n,
        "waves": waves,
        "elapsed_s": round(elapsed, 3),
        "throughput_fps": round(n / elapsed, 2) if elapsed > 0 else 0.0,
        "cancelled": cancelled,
        "batch_trajectory": trajectory,
        "scheduler_decisions": decisions,
        "scheduler_converged": sched.converged,
        "bottleneck_summary": {
            "preprocess_ms_total": round(total_preprocess_ms, 1),
            "forward_ms_total": round(total_forward_ms, 1),
            "postprocess_ms_total": round(total_postprocess_ms, 1),
            "gpu_util_avg": round(sum(util_samples) / len(util_samples), 1)
            if util_samples
            else None,
            "vram_used_peak_mb": round(vram_peak, 1),
            "vram_after_cleanup_mb": round(after.allocated_mb, 1),
        },
        "frames": [
            {
                "frame_index": f.frame_index,
                "time_sec": f.time_sec,
                "n_objects": f.n_objects,
                "detections": f.detections,
            }
            for f in frames_out
        ],
        "frame_objects": list(frames_out) if keep_raw else None,
    }


def extract_frames_to_list(
    video_path: str,
    *,
    sample_fps: float | None = None,
    stride: int = 1,
    start_s: float = 0.0,
    end_s: float | None = None,
    max_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Stream-extract frame metadata (index, timestamp, size). Does not hold all frames."""
    out: list[dict[str, Any]] = []
    for fi, ts, img in _stream_frames(
        video_path,
        sample_fps=sample_fps,
        stride=stride,
        start_s=start_s,
        end_s=end_s,
        max_frames=max_frames,
    ):
        out.append(
            {"frame_index": fi, "time_sec": round(ts, 4), "width": img.width, "height": img.height}
        )
    return out


__all__ = ["CancelToken", "FrameResult", "extract_frames_to_list", "infer_video"]
