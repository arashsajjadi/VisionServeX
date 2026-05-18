# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Persistent-load detection benchmark with GPU enforcement (v2.17.0).

The v16/v19 notebook reported seconds-per-image D-FINE / RF-DETR latency.
That smelled like CPU fallback, per-image model reload, or a subprocess
per image. This module produces a benchmark run that *proves* the
opposite by tracking:

- ``load_count``: must be 1 for N images.
- ``device_actual``: must equal ``device_requested`` when ``require_gpu``
  is set; otherwise the run fails with ``GPU_REQUIRED_NOT_USED``.
- A timing breakdown (preprocess / inference / postprocess / evaluation),
  not a single opaque latency.
- An optional GPU utilization sampler (``--sample-gpu``) that polls
  ``nvidia-smi`` in a background thread.

The schema is documented in the docstring of :func:`run_persistent_detection_benchmark`.
"""

from __future__ import annotations

import contextlib
import math
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "GpuUtilizationSample",
    "GpuUtilizationSampler",
    "build_v217_row",
    "run_persistent_detection_benchmark",
]


# ---------------------------------------------------------------------------
# GPU sampler
# ---------------------------------------------------------------------------


@dataclass
class GpuUtilizationSample:
    """One snapshot from `nvidia-smi`."""

    timestamp: float
    gpu_utilization_pct: float  # 0.0 - 100.0
    vram_used_mb: float
    vram_total_mb: float


@dataclass
class GpuUtilizationSampler:
    """Background ``nvidia-smi`` sampler.

    Call :meth:`start` before the benchmark loop and :meth:`stop` after.
    :meth:`summary` returns a dict with ``samples`` / ``utilization_mean`` /
    ``utilization_p50`` / ``utilization_p95`` / ``vram_used_peak_gb`` /
    ``vram_used_mean_gb``. If ``nvidia-smi`` is not on PATH, the sampler
    silently records zero samples and emits a warning. It never raises.
    """

    interval: float = 0.5
    _samples: list[GpuUtilizationSample] = field(default_factory=list)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _warning: str = ""

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            self._warning = "nvidia-smi not on PATH; GPU sampling disabled"
            return
        self._thread = threading.Thread(target=self._run, name="vsx-gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 4))

    def summary(self) -> dict[str, Any]:
        utils = [s.gpu_utilization_pct for s in self._samples]
        vram = [s.vram_used_mb for s in self._samples]

        def _pct(xs: list[float], q: float) -> float:
            if not xs:
                return 0.0
            xs_sorted = sorted(xs)
            return float(xs_sorted[min(len(xs_sorted) - 1, int(len(xs_sorted) * q))])

        peak_mb = max(vram) if vram else 0.0
        mean_mb = sum(vram) / len(vram) if vram else 0.0
        out = {
            "samples": len(self._samples),
            "utilization_mean": round(sum(utils) / len(utils), 2) if utils else 0.0,
            "utilization_p50": round(_pct(utils, 0.50), 2),
            "utilization_p95": round(_pct(utils, 0.95), 2),
            "vram_used_peak_gb": round(peak_mb / 1024.0, 3),
            "vram_used_mean_gb": round(mean_mb / 1024.0, 3),
            "interval_s": self.interval,
            "warnings": [self._warning] if self._warning else [],
        }
        return out

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )
                line = (out.stdout or "").strip().splitlines()
                if line:
                    parts = [p.strip() for p in line[0].split(",")]
                    if len(parts) >= 3:
                        self._samples.append(
                            GpuUtilizationSample(
                                timestamp=time.time(),
                                gpu_utilization_pct=float(parts[0]),
                                vram_used_mb=float(parts[1]),
                                vram_total_mb=float(parts[2]),
                            )
                        )
            except Exception:
                # Sampler must never raise; record nothing for this tick.
                pass
            self._stop_event.wait(self.interval)


# ---------------------------------------------------------------------------
# v2.17.0 row builder
# ---------------------------------------------------------------------------


def build_v217_row(
    *,
    model_id: str,
    canonical_model_id: str,
    is_alias: bool,
    n_images_requested: int,
    n_images_evaluated: int,
    device_requested: str,
    device_actual: str,
    gpu_name: str,
    gpu_profile: str,
    load_count: int,
    load_time_ms: float,
    preprocess_ms: list[float],
    inference_ms: list[float],
    postprocess_ms: list[float],
    evaluation_ms: list[float],
    total_latency_ms: list[float],
    n_raw_predictions: int,
    n_normalized_predictions: int,
    n_invalid_predictions: int,
    n_dropped_predictions: int,
    no_detection_image_count: int,
    ap50: float | None,
    ap75: float | None,
    map50_95: float | None,
    class_agnostic_ap50: float | None,
    precision50: float | None,
    recall50: float | None,
    f1_50: float | None,
    gpu_utilization: dict[str, Any] | None,
    warnings_: list[str],
    errors: list[str],
    status: str,
    code: str,
    evaluation_scope: str | None = None,
) -> dict[str, Any]:
    """Assemble the v2.17.0 detection benchmark row shape.

    `evaluation_scope` follows :func:`runtime.leaderboard.classify_row` rules:
    ``failed``, ``diagnostic_6``, ``diagnostic_partial``, or ``full_<N>``.
    """
    if evaluation_scope is None:
        if n_images_evaluated == 0:
            evaluation_scope = "failed"
        elif n_images_evaluated <= 6 and n_images_requested > 6:
            evaluation_scope = "diagnostic_6"
        elif n_images_evaluated == n_images_requested:
            evaluation_scope = f"full_{n_images_requested}"
        elif n_images_evaluated < n_images_requested:
            evaluation_scope = "diagnostic_partial"
        else:
            evaluation_scope = f"full_{n_images_evaluated}"

    def _p(xs: list[float], q: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        return float(s[min(len(s) - 1, int(len(s) * q))])

    def _mean(xs: list[float]) -> float:
        return float(statistics.fmean(xs)) if xs else 0.0

    total_p50 = _p(total_latency_ms, 0.50)
    total_p95 = _p(total_latency_ms, 0.95)
    ips = (1000.0 / total_p50) if total_p50 > 0 else 0.0

    return {
        "model_id": model_id,
        "canonical_model_id": canonical_model_id,
        "is_alias": is_alias,
        "n_images_requested": n_images_requested,
        "n_images_evaluated": n_images_evaluated,
        "n_images": n_images_evaluated,  # back-compat alias
        "evaluation_scope": evaluation_scope,
        "device_requested": device_requested,
        "device_actual": device_actual,
        "gpu_name": gpu_name,
        "gpu_profile": gpu_profile,
        "load_count": load_count,
        "load_time_ms": round(load_time_ms, 2),
        "preprocess_ms_p50": round(_p(preprocess_ms, 0.50), 2),
        "inference_ms_p50": round(_p(inference_ms, 0.50), 2),
        "postprocess_ms_p50": round(_p(postprocess_ms, 0.50), 2),
        "evaluation_ms_p50": round(_p(evaluation_ms, 0.50), 2),
        "total_latency_ms_p50": round(total_p50, 2),
        "total_latency_ms_p95": round(total_p95, 2),
        "latency_p50_ms": round(total_p50, 2),  # back-compat
        "latency_p95_ms": round(total_p95, 2),  # back-compat
        "preprocess_ms_mean": round(_mean(preprocess_ms), 2),
        "inference_ms_mean": round(_mean(inference_ms), 2),
        "postprocess_ms_mean": round(_mean(postprocess_ms), 2),
        "images_per_second": round(ips, 2),
        "n_raw_predictions": n_raw_predictions,
        "n_normalized_predictions": n_normalized_predictions,
        "n_invalid_predictions": n_invalid_predictions,
        "n_dropped_predictions": n_dropped_predictions,
        "no_detection_image_count": no_detection_image_count,
        "ap50": _round_or_none(ap50),
        "ap75": _round_or_none(ap75),
        "map50_95": _round_or_none(map50_95),
        "class_agnostic_ap50": _round_or_none(class_agnostic_ap50),
        "precision50": _round_or_none(precision50),
        "recall50": _round_or_none(recall50),
        "f1_50": _round_or_none(f1_50),
        "gpu_utilization": gpu_utilization,
        "warnings": list(warnings_),
        "errors": list(errors),
        "status": status,
        "code": code,
        "task": "detect",
    }


def _round_or_none(v: float | None) -> float | None:
    if v is None:
        return None
    try:
        if math.isnan(float(v)):
            return None
    except (TypeError, ValueError):
        return None
    return round(float(v), 4)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_persistent_detection_benchmark(
    *,
    model_id: str,
    samples: list,  # list[DatasetSample] from runtime.evaluation
    device_requested: str,
    require_gpu: bool,
    sample_gpu: bool,
    gpu_sample_interval: float = 0.5,
    dataset_name: str = "user_dataset",
) -> dict[str, Any]:
    """Run one detection model on a labelled dataset with persistent load.

    The model is loaded ONCE before the loop and ONCE after; `load_count`
    in the returned dict is 1 for a healthy run. Per-image timing is split
    into preprocess / inference / postprocess / evaluation. Failures emit
    a structured row (``status=failed``, ``code=...``) rather than a
    raw exception.

    Args:
        model_id: registry model ID (use a canonical one, not an alias).
        samples: a list of :class:`runtime.evaluation.DatasetSample`.
        device_requested: ``cuda``, ``cpu``, ``auto``, etc.
        require_gpu: if True, a CPU fallback fails the run with
            ``GPU_REQUIRED_NOT_USED``.
        sample_gpu: if True, spawn a background nvidia-smi sampler.
        gpu_sample_interval: seconds between samples.
        dataset_name: cosmetic, written into the row.
    """
    from visionservex.core.model import VisionModel
    from visionservex.core.results import DetectionResult
    from visionservex.engines.base import MissingDependencyError
    from visionservex.runtime.downloads import DownloadError, ManualDownloadRequired
    from visionservex.runtime.evaluation import DetectionEvaluator
    from visionservex.runtime.gpu_profile import detect_gpu_profile
    from visionservex.runtime.leaderboard import canonicalize_model_id

    canonical, is_alias = canonicalize_model_id(model_id)
    gpu_info = detect_gpu_profile()
    gpu_name = gpu_info.gpu_name or ""
    gpu_profile = gpu_info.profile

    n_requested = len(samples)
    warnings_: list[str] = []
    errors: list[str] = []

    # ----- Load model (load_count tracking) -----
    load_count = 0
    t_load_start = time.perf_counter()
    try:
        model = VisionModel(model_id, device=device_requested)
        model._ensure_loaded()
        load_count = 1
        load_time_ms = (time.perf_counter() - t_load_start) * 1000.0
    except (MissingDependencyError, ManualDownloadRequired, DownloadError) as exc:
        return build_v217_row(
            model_id=model_id,
            canonical_model_id=canonical,
            is_alias=is_alias,
            n_images_requested=n_requested,
            n_images_evaluated=0,
            device_requested=device_requested,
            device_actual="",
            gpu_name=gpu_name,
            gpu_profile=gpu_profile,
            load_count=0,
            load_time_ms=(time.perf_counter() - t_load_start) * 1000.0,
            preprocess_ms=[],
            inference_ms=[],
            postprocess_ms=[],
            evaluation_ms=[],
            total_latency_ms=[],
            n_raw_predictions=0,
            n_normalized_predictions=0,
            n_invalid_predictions=0,
            n_dropped_predictions=0,
            no_detection_image_count=0,
            ap50=None,
            ap75=None,
            map50_95=None,
            class_agnostic_ap50=None,
            precision50=None,
            recall50=None,
            f1_50=None,
            gpu_utilization=None,
            warnings_=warnings_,
            errors=[str(exc)[:300]],
            status="expected_blocker",
            code="DEPENDENCY_REQUIRED",
            evaluation_scope="failed",
        )
    except Exception as exc:
        return build_v217_row(
            model_id=model_id,
            canonical_model_id=canonical,
            is_alias=is_alias,
            n_images_requested=n_requested,
            n_images_evaluated=0,
            device_requested=device_requested,
            device_actual="",
            gpu_name=gpu_name,
            gpu_profile=gpu_profile,
            load_count=0,
            load_time_ms=(time.perf_counter() - t_load_start) * 1000.0,
            preprocess_ms=[],
            inference_ms=[],
            postprocess_ms=[],
            evaluation_ms=[],
            total_latency_ms=[],
            n_raw_predictions=0,
            n_normalized_predictions=0,
            n_invalid_predictions=0,
            n_dropped_predictions=0,
            no_detection_image_count=0,
            ap50=None,
            ap75=None,
            map50_95=None,
            class_agnostic_ap50=None,
            precision50=None,
            recall50=None,
            f1_50=None,
            gpu_utilization=None,
            warnings_=warnings_,
            errors=[str(exc)[:300]],
            status="failed",
            code="MODEL_LOAD_FAILED",
            evaluation_scope="failed",
        )

    device_actual = str(getattr(model, "device", device_requested))

    # GPU enforcement
    if require_gpu and not device_actual.startswith("cuda"):
        with contextlib.suppress(Exception):
            model.close()
        return build_v217_row(
            model_id=model_id,
            canonical_model_id=canonical,
            is_alias=is_alias,
            n_images_requested=n_requested,
            n_images_evaluated=0,
            device_requested=device_requested,
            device_actual=device_actual,
            gpu_name=gpu_name,
            gpu_profile=gpu_profile,
            load_count=load_count,
            load_time_ms=load_time_ms,
            preprocess_ms=[],
            inference_ms=[],
            postprocess_ms=[],
            evaluation_ms=[],
            total_latency_ms=[],
            n_raw_predictions=0,
            n_normalized_predictions=0,
            n_invalid_predictions=0,
            n_dropped_predictions=0,
            no_detection_image_count=0,
            ap50=None,
            ap75=None,
            map50_95=None,
            class_agnostic_ap50=None,
            precision50=None,
            recall50=None,
            f1_50=None,
            gpu_utilization=None,
            warnings_=warnings_,
            errors=[
                (
                    f"--require-gpu was set but device_actual={device_actual!r}; "
                    "the model fell back to CPU. Refusing to benchmark."
                )
            ],
            status="failed",
            code="GPU_REQUIRED_NOT_USED",
            evaluation_scope="failed",
        )

    # ----- Optional GPU sampler -----
    sampler = GpuUtilizationSampler(interval=gpu_sample_interval) if sample_gpu else None
    if sampler is not None:
        sampler.start()

    # ----- Inference loop -----
    evaluator = DetectionEvaluator()
    class_agnostic_evaluator = DetectionEvaluator()

    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    evaluation_ms: list[float] = []
    total_latency_ms: list[float] = []

    from PIL import Image as _PIL

    n_evaluated = 0
    n_raw = 0
    n_normalized = 0
    n_invalid = 0
    n_dropped = 0
    n_no_det = 0

    try:
        for sample in samples:
            t_total = time.perf_counter()
            try:
                t_pre = time.perf_counter()
                img = _PIL.open(sample.image_path).convert("RGB")
                w, h = img.size
                preprocess_ms.append((time.perf_counter() - t_pre) * 1000.0)

                t_inf = time.perf_counter()
                result = model.predict(img, threshold=0.001)
                inference_ms.append((time.perf_counter() - t_inf) * 1000.0)

                t_post = time.perf_counter()
                if isinstance(result, DetectionResult):
                    dets = result.detections
                    n_raw += len(dets)
                    pred_boxes = []
                    pred_scores = []
                    pred_classes = []
                    pred_classes_agnostic = []
                    for d in dets:
                        b = d.box
                        if (
                            b.x1 < 0
                            or b.y1 < 0
                            or b.x2 > w
                            or b.y2 > h
                            or b.x1 >= b.x2
                            or b.y1 >= b.y2
                        ):
                            n_invalid += 1
                            continue
                        pred_boxes.append([b.x1, b.y1, b.x2, b.y2])
                        pred_scores.append(d.score)
                        pred_classes.append(d.label)
                        pred_classes_agnostic.append("object")
                    n_normalized += len(pred_boxes)
                    if not pred_boxes:
                        n_no_det += 1
                else:
                    pred_boxes = []
                    pred_scores = []
                    pred_classes = []
                    pred_classes_agnostic = []
                    n_no_det += 1
                postprocess_ms.append((time.perf_counter() - t_post) * 1000.0)

                t_eval = time.perf_counter()
                evaluator.add_image(
                    pred_boxes,
                    pred_scores,
                    pred_classes,
                    sample.gt_boxes,
                    sample.gt_classes,
                )
                class_agnostic_evaluator.add_image(
                    pred_boxes,
                    pred_scores,
                    pred_classes_agnostic,
                    sample.gt_boxes,
                    ["object"] * len(sample.gt_classes),
                )
                evaluation_ms.append((time.perf_counter() - t_eval) * 1000.0)
                total_latency_ms.append((time.perf_counter() - t_total) * 1000.0)
                n_evaluated += 1
            except Exception as exc:
                errors.append(f"image {sample.image_path}: {exc!s:.200}")
                n_dropped += 1
                continue
    finally:
        if sampler is not None:
            sampler.stop()
        with contextlib.suppress(Exception):
            model.close()

    if n_evaluated == 0:
        return build_v217_row(
            model_id=model_id,
            canonical_model_id=canonical,
            is_alias=is_alias,
            n_images_requested=n_requested,
            n_images_evaluated=0,
            device_requested=device_requested,
            device_actual=device_actual,
            gpu_name=gpu_name,
            gpu_profile=gpu_profile,
            load_count=load_count,
            load_time_ms=load_time_ms,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            evaluation_ms=evaluation_ms,
            total_latency_ms=total_latency_ms,
            n_raw_predictions=n_raw,
            n_normalized_predictions=n_normalized,
            n_invalid_predictions=n_invalid,
            n_dropped_predictions=n_dropped,
            no_detection_image_count=n_no_det,
            ap50=None,
            ap75=None,
            map50_95=None,
            class_agnostic_ap50=None,
            precision50=None,
            recall50=None,
            f1_50=None,
            gpu_utilization=(sampler.summary() if sampler is not None else None),
            warnings_=warnings_,
            errors=errors or ["no images evaluated"],
            status="failed",
            code="NO_IMAGES_EVALUATED",
            evaluation_scope="failed",
        )

    # ----- Metrics (class-aware) -----
    metrics = evaluator.compute_map50_95()
    metrics_75 = evaluator.compute_metrics(iou_threshold=0.75)

    # Class-agnostic AP50
    cag = class_agnostic_evaluator.compute_metrics(iou_threshold=0.50)

    gpu_summary = sampler.summary() if sampler is not None else None
    if sampler is not None and gpu_summary is not None:
        warnings_.extend(gpu_summary.get("warnings", []))

    return build_v217_row(
        model_id=model_id,
        canonical_model_id=canonical,
        is_alias=is_alias,
        n_images_requested=n_requested,
        n_images_evaluated=n_evaluated,
        device_requested=device_requested,
        device_actual=device_actual,
        gpu_name=gpu_name,
        gpu_profile=gpu_profile,
        load_count=load_count,
        load_time_ms=load_time_ms,
        preprocess_ms=preprocess_ms,
        inference_ms=inference_ms,
        postprocess_ms=postprocess_ms,
        evaluation_ms=evaluation_ms,
        total_latency_ms=total_latency_ms,
        n_raw_predictions=n_raw,
        n_normalized_predictions=n_normalized,
        n_invalid_predictions=n_invalid,
        n_dropped_predictions=n_dropped,
        no_detection_image_count=n_no_det,
        ap50=metrics.get("map50"),
        ap75=metrics_75.get("map50"),  # mAP at IoU=0.75
        map50_95=metrics.get("map50_95"),
        class_agnostic_ap50=cag.get("map50"),
        precision50=metrics.get("precision"),
        recall50=metrics.get("recall"),
        f1_50=metrics.get("f1"),
        gpu_utilization=gpu_summary,
        warnings_=warnings_,
        errors=errors,
        status="ok",
        code="OK",
    )


def _dummy_close() -> None:
    """Type stub so VisionModel.close fallbacks compile cleanly in tests."""
    return None
