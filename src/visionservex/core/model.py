# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""High-level ``VisionModel`` API — Ultralytics-like ergonomics.

A ``VisionModel`` is the friendly facade users interact with. It hides
engine selection, device choice, weight download, and result construction
behind a single object.

Ultralytics-like workflow::

    model = VisionModel("dfine-x-o365-coco")
    model.pull()
    model.info()
    results = model.predict("image.jpg", conf=0.25)
    results.save("outputs/")
    results.plot()
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from visionservex.config import get_settings
from visionservex.core.results import BaseResult
from visionservex.engines.base import BaseEngine
from visionservex.engines.registry import build_engine
from visionservex.registry import ModelEntry, default_registry
from visionservex.runtime.device import resolve_device
from visionservex.runtime.downloads import (
    DownloadError,
    cached_path,
    download,
    is_cached,
)
from visionservex.utils.images import open_safe
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)

# Task groups for the task-specific public API (v3.17.0).
_CLASSIFY_TASKS = frozenset({"classify", "classification"})
_EMBED_TASKS = frozenset({"embed", "embedding"})
_SEGMENT_TASKS = frozenset({"foundation_segment", "segment", "grounded_segment"})
_DETECT_TASKS = frozenset({"detect", "obb", "open_vocab_detect"})
_OPEN_VOCAB_TASKS = frozenset({"open_vocab_detect"})

# v3.21: model-family -> isolated Docker sidecar that can serve it when the host
# environment cannot. A family appearing here means a sidecar *exists*; whether it
# is *live-verified* is governed independently by ``live_evidence.LIVE_SIDECAR_VERIFIED``.
_SIDECAR_BY_FAMILY: dict[str, str] = {
    "florence-2": "florence2",
    "rtmdet": "openmmlab",
    "rtmpose": "openmmlab",
    "deim": "deimv2",
    "deimv2": "deimv2",
    "rtdetrv4": "rtdetrv4",
}


# HF ``SamModel`` engines whose mask decoder can be fine-tuned with frozen
# encoders (``training.segmentation_finetune``). SAM2 uses a different arch.
_SAM_DECODER_FINETUNE_ENGINES = frozenset({"sam_hf"})


def _fine_tune_kind(*, train_ready: bool, task: str, inference_ready: bool, engine: str) -> str:
    """Classify the *kind* of fine-tune a model honestly supports (v3.21).

    - ``full_supervised``         — end-to-end trainable detector/classifier
      (RF-DETR, LibreYOLO, torchvision classifiers).
    - ``frozen_backbone_head``    — frozen-backbone head fine-tune for embedding
      backbones, linear probe OR deeper MLP head
      (``training.embedding_finetune``, ``head_type='linear'|'mlp'``).
    - ``frozen_encoder_decoder``  — frozen-encoder SAM mask-decoder fine-tune for
      HF SamModel segmenters (``training.segmentation_finetune``).
    - ``none``                    — no fine-tune path wired.
    """
    if train_ready:
        return "full_supervised"
    if task in _EMBED_TASKS and inference_ready:
        return "frozen_backbone_head"
    if task in _SEGMENT_TASKS and engine in _SAM_DECODER_FINETUNE_ENGINES and inference_ready:
        return "frozen_encoder_decoder"
    return "none"


def _require_task(entry: ModelEntry, allowed: frozenset[str], method: str) -> None:
    if entry.task not in allowed:
        from visionservex.exceptions import TaskNotSupportedError

        raise TaskNotSupportedError(
            entry.id,
            method,
            entry.task,
            hint=f"{method}() requires task in {sorted(allowed)}; use predict() instead.",
        )


def list_models(task: str | None = None, *, family: str | None = None) -> list[str]:
    """Return every registered model id (optionally filtered by task/family)."""
    return sorted(e.id for e in default_registry().list(task=task, family=family))


# v3.22.0 — segmentation output serialization flags (Phase 5).
_SEG_FLAG_NAMES = (
    "return_boxes",
    "return_masks",
    "return_rle",
    "return_polygons",
    "return_quality",
    "max_polygon_points",
    "polygon_simplification_tolerance",
)


def _pop_seg_flags(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove segmentation serialization flags from kwargs (don't pass to engines)."""
    return {k: kwargs.pop(k) for k in _SEG_FLAG_NAMES if k in kwargs}


def _apply_seg_flags(result: BaseResult, flags: dict[str, Any]) -> None:
    """Apply serialization flags to a SegmentationResult so to_dict honors them."""
    if not flags:
        return
    from visionservex.core.results import SegmentationResult

    if isinstance(result, SegmentationResult):
        for k, v in flags.items():
            setattr(result, k, v)


class VisionModel:
    """User-facing wrapper around an engine and a registry entry.

    Example::

        model = VisionModel("mock-detect")
        result = model.predict("image.jpg")
        result.save("out.jpg")

    Auto-pull example (Python only; server is a separate config path)::

        model = VisionModel("grounding-dino-tiny", auto_pull=True)
        result = model.predict("image.jpg", prompts=["cat", "dog"])
    """

    def __init__(
        self,
        model_id: str,
        *,
        task: str | None = None,
        device: str | None = None,
        precision: str | None = None,
        auto_pull: bool = False,
    ) -> None:
        self.settings = get_settings()
        self.entry: ModelEntry = default_registry().get(model_id)
        if task is not None and task != self.entry.task:
            raise ValueError(
                f"model {model_id!r} is registered for task {self.entry.task!r}, not {task!r}"
            )

        chosen_device = resolve_device(
            preference=device or self.settings.runtime.device_preference,
            supported=self.entry.supported_devices,
        )
        chosen_precision = self._resolve_precision(precision, chosen_device)
        self.device = chosen_device
        self.precision = chosen_precision
        self.auto_pull = auto_pull

        self.engine: BaseEngine = build_engine(self.entry)
        self._loaded = False
        self._cache_path: str | None = None
        self._model_loaded_from: str | None = None

    # ------- properties -------

    def _resolve_precision(self, precision: str | None, device: str) -> str:
        pref = precision or self.settings.runtime.precision_preference
        if pref == "auto":
            # Default to fp32 everywhere — HF models that use text encoders
            # (Grounding DINO, etc.) silently mix integer and float tensors which
            # breaks if we force fp16 without also casting the model weights.
            # Use fp32 for safety; advanced users can explicitly pass precision="fp16".
            return "fp32"
        if pref in self.entry.supported_precisions:
            return pref
        return self.entry.supported_precisions[0]

    # ------- lifecycle -------

    def warmup(self) -> None:
        """Load weights eagerly and run a tiny dummy inference if supported."""
        self._ensure_loaded()
        self.engine.warmup()

    def unload(self) -> None:
        """Unload the model and flush GPU caches.

        Performs the full cleanup sequence:
        1. Engine.unload() (drops HF/torch model references).
        2. Python garbage collection.
        3. CUDA cache flush (empty_cache + ipc_collect + reset_peak_stats).

        This prevents stepwise VRAM accumulation across sequential model loads.
        """
        if self._loaded:
            self.engine.unload()
            self._loaded = False
        from visionservex.runtime.gpu_lifecycle import clear_torch_cuda_cache, force_gc

        force_gc()
        clear_torch_cuda_cache()

    def close(self) -> None:
        """Alias for unload(). Mirrors the Ultralytics-style .close() idiom."""
        self.unload()

    def _ensure_weights(self) -> None:
        """Ensure local weights exist; pull if allowed."""
        if self.entry.download_type == "synthetic":
            return
        if is_cached(self.entry):
            cp = cached_path(self.entry)
            self._cache_path = str(cp) if cp else None
            self._model_loaded_from = "cache"
            return
        if not self.auto_pull:
            # Engine.load() will surface an error; we pre-attach a helpful hint here
            # so the user sees the pull command without digging through tracebacks.
            if not is_cached(self.entry):
                import warnings

                warnings.warn(
                    f"Checkpoint for '{self.entry.id}' is not cached. "
                    f"Run: visionservex model pull {self.entry.id}  "
                    "(or pass auto_pull=True / --auto-pull)",
                    stacklevel=4,
                )
            return
        try:
            path = download(self.entry)
            self._cache_path = str(path)
            self._model_loaded_from = self.entry.download_type
        except DownloadError as exc:
            raise RuntimeError(f"could not auto-pull weights for {self.entry.id!r}: {exc}") from exc

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._ensure_weights()
            self.engine.load(device=self.device, precision=self.precision)
            self._loaded = True

    def __enter__(self) -> VisionModel:
        self._ensure_loaded()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unload()

    # ------- inference -------

    @property
    def loaded(self) -> bool:
        """True when model weights are in memory."""
        return self._loaded

    def predict(
        self,
        image: Image.Image | bytes | str | Path,
        *,
        prompts: Sequence[str] | None = None,
        prompt: str | None = None,
        box: list[float] | None = None,
        boxes: list[list[float]] | None = None,
        points: list[list[float]] | None = None,
        point_labels: list[int] | None = None,
        labels: list[int] | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
        task: str | None = None,
        unload_after: bool = False,
        **kwargs: Any,
    ) -> BaseResult:
        """Run inference.

        Convenience aliases so callers never need to know backend details:

        - ``prompt="car, person"``    → split into prompts list
        - ``box=[x1,y1,x2,y2]``       → passed as ``boxes=[[...]]``
        - ``points=[[x,y]]``          → point prompts for SAM-style models
        - ``labels=[1]`` / ``point_labels=[1]``  → foreground labels
        - ``top_k=5``                 → classification top-k
        - ``threshold=0.3``           → detection score threshold
        - ``task="semantic"``         → OneFormer task override
        """
        # Normalise convenience aliases
        if prompt is not None and not prompts:
            prompts = [p.strip() for p in prompt.split(",") if p.strip()]
        if box is not None and boxes is None:
            boxes = [box]
        if labels is not None and point_labels is None:
            point_labels = labels

        # Forward relevant kwargs to engine
        if boxes is not None:
            kwargs["boxes"] = boxes
        if points is not None:
            kwargs["points"] = points
        if point_labels is not None:
            kwargs["point_labels"] = point_labels
        if top_k is not None:
            kwargs["top_k"] = top_k
        if threshold is not None:
            kwargs["threshold"] = threshold
        if task is not None:
            kwargs["task"] = task

        seg_flags = _pop_seg_flags(kwargs)
        self._ensure_loaded()
        pil = self._coerce_image(image)
        start = time.perf_counter()
        result = self.engine.predict(pil, prompts=prompts, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._finalize_result(result, pil, latency_ms)
        _apply_seg_flags(result, seg_flags)
        if unload_after:
            self.unload()
        return result

    def _finalize_result(
        self, result: BaseResult, pil: Image.Image, latency_ms: float
    ) -> BaseResult:
        """Attach common provenance/timing fields to an engine result."""
        result.latency_ms = latency_ms
        result.model_id = self.entry.id
        result.task = self.entry.task
        result.device = self.device
        result.precision = self.precision
        result.backend = (
            getattr(self.engine, "backend_label", self.entry.backend) or self.entry.backend
        )
        result.model_loaded_from = self._model_loaded_from
        result.cache_path = self._cache_path
        result.image_size = pil.size
        result._image = pil
        return result

    @property
    def supports_true_batch(self) -> bool:
        """True only if the engine runs ONE forward over a stacked batch (N>1)."""
        return bool(getattr(self.engine, "supports_true_batch", False))

    @property
    def max_batch_size_hint(self) -> int:
        return int(getattr(self.engine, "max_batch_size_hint", 1))

    @property
    def preferred_batch_sizes(self) -> tuple[int, ...]:
        return tuple(getattr(self.engine, "preferred_batch_sizes", (1,)))

    def batch_predict(
        self,
        images: Iterable[Image.Image | bytes | str | Path],
        *,
        prompts: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[BaseResult]:
        """Run inference over a list of images.

        Delegates to ``engine.predict_batch`` — which is a TRUE tensor batch
        (one forward over a stacked batch) for engines that set
        ``supports_true_batch=True`` (e.g. D-FINE), and an HONEST internal loop
        otherwise. Each result's ``metadata['batch_mode']`` reports which path
        was taken; the worker NEVER labels a loop as a true batch.
        """
        seg_flags = _pop_seg_flags(kwargs)
        self._ensure_loaded()
        pil_images = [self._coerce_image(img) for img in images]
        if not pil_images:
            return []
        start = time.perf_counter()
        results = self.engine.predict_batch(pil_images, prompts=prompts, **kwargs)
        total_ms = (time.perf_counter() - start) * 1000.0
        share = total_ms / max(1, len(results))
        for r, pil in zip(results, pil_images, strict=False):
            self._finalize_result(r, pil, share)
            _apply_seg_flags(r, seg_flags)
        return results

    def stream(
        self,
        images: Iterable[Image.Image | bytes | str | Path],
        **kwargs: Any,
    ):
        for img in images:
            yield self.predict(img, **kwargs)

    # ------- introspection / utilities -------

    def info(self) -> dict[str, Any]:
        return {
            "id": self.entry.id,
            "task": self.entry.task,
            "device": self.device,
            "precision": self.precision,
            "engine": self.entry.engine,
            "backend": self.entry.backend,
            "status": self.entry.status,
            "implementation_status": self.entry.implementation_status,
            "license": self.entry.license,
            "loaded": self._loaded,
            "cache_path": self._cache_path,
            "auto_pull": self.auto_pull,
        }

    def export(self, format: str, output_path: str | Path) -> Path:
        return self.engine.export(format, Path(output_path))

    def benchmark(self, image: Image.Image | bytes | str | Path, *, n: int = 5) -> dict[str, Any]:
        pil = self._coerce_image(image)
        self._ensure_loaded()
        self.engine.warmup()
        latencies = []
        for _ in range(max(1, n)):
            t0 = time.perf_counter()
            self.engine.predict(pil)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        latencies.sort()
        return {
            "n": len(latencies),
            "p50_ms": latencies[len(latencies) // 2],
            "p90_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.9))],
            "p99_ms": latencies[-1],
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
            "device": self.device,
            "model_id": self.entry.id,
            "backend": self.entry.backend,
        }

    # ------- Ultralytics-like API extensions -------

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> VisionModel:
        """Alias for VisionModel(model_id). Ultralytics-style factory."""
        return cls(model_id, **kwargs)

    @classmethod
    def from_registry(cls, model_id: str, **kwargs: Any) -> VisionModel:
        """Alias for VisionModel(model_id). Explicit registry-based factory."""
        return cls(model_id, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        model_id: str,
        device: str | None = None,
        **kwargs: Any,
    ) -> VisionModel:
        """Load a trained checkpoint for inference and return a ready model.

        ``model_id`` identifies the family/variant the checkpoint was trained for
        (e.g. ``"libreyolo-yolox-s"``) — it is required because a bare ``.pt``
        does not reliably carry the engine/variant. The returned model predicts
        through the SAME normalized schema as base inference, using the trained
        weights with no base-weight fallback.

        Supported for engines that implement ``load_checkpoint`` (LibreYOLO,
        RF-DETR). Others raise a structured :class:`NotImplementedError`.

        Example::

            res = VisionModel("libreyolo-rtdetr-r50").train("data.yaml", epochs=1)
            m = VisionModel.from_checkpoint(res["best_checkpoint"],
                                            model_id="libreyolo-rtdetr-r50", device="cuda")
            pred = m.predict(image)
        """
        model = cls(model_id, device=device, **kwargs)
        model.load_checkpoint(checkpoint_path, device=device)
        return model

    def load_checkpoint(
        self, checkpoint_path: str | Path, *, device: str | None = None
    ) -> VisionModel:
        """Load a trained checkpoint into this model in place (for inference).

        Delegates to the engine's ``load_checkpoint``. After this call,
        :meth:`predict` uses the trained weights — there is no silent reload of
        the base weights. Raises :class:`NotImplementedError` for engines that
        do not support trained-checkpoint reload.
        """
        fn = getattr(self.engine, "load_checkpoint", None)
        if fn is None:
            raise NotImplementedError(
                f"CHECKPOINT_LOAD_UNSUPPORTED: the {self.entry.family!r} engine does not "
                "support trained-checkpoint reload. Supported families: LibreYOLO, RF-DETR."
            )
        dev = device or self.device
        fn(checkpoint_path, device=dev)
        self.device = dev
        self._loaded = True
        return self

    def to(self, device: str) -> VisionModel:
        """Move model to a device (Ultralytics-style). Returns self."""
        self.device = device
        if self._loaded:
            self.unload()
        return self

    def pull(self, *, force: bool = False) -> Path | None:
        """Download model weights. Returns local path or None for synthetic models."""
        from visionservex.runtime.downloads import download

        if self.entry.download_type == "synthetic":
            return None
        path = download(self.entry, force=force)
        self._cache_path = str(path)
        self._model_loaded_from = self.entry.download_type
        return path

    def cache_info(self) -> dict[str, Any]:
        """Return cache metadata for this model."""
        from visionservex.config import get_settings
        from visionservex.runtime.downloads import cached_path, is_cached

        cp = cached_path(self.entry)
        size_bytes = 0
        if cp and cp.exists():
            if cp.is_dir():
                size_bytes = sum(f.stat().st_size for f in cp.rglob("*") if f.is_file())
            else:
                size_bytes = cp.stat().st_size
        return {
            "model_id": self.entry.id,
            "cached": is_cached(self.entry),
            "cache_path": str(cp) if cp else None,
            "size_bytes": size_bytes,
            "cache_dir": str(get_settings().cache.cache_dir),
            "auto_download": self.entry.auto_download,
        }

    def checkpoint_info(self) -> dict[str, Any]:
        """Return checkpoint provenance and trust metadata."""
        return {
            "model_id": self.entry.id,
            "source": self.entry.download_type,
            "hf_repo_id": getattr(self.entry, "hf_repo_id", None),
            "upstream_url": self.entry.upstream_url,
            "license": self.entry.license,
            "license_uncertain": self.entry.license_uncertain or False,
            "implementation_status": self.entry.implementation_status,
            "checkpoint_source": self.entry.download_type,
            "checkpoint_trust_level": (
                "community_hf"
                if self.entry.download_type == "huggingface"
                else "package_managed"
                if self.entry.download_type == "package_managed"
                else "manual"
            ),
            "official_ap_claim": "see model-card for upstream benchmark claims",
            "verified_by_visionservex": "latency_tested_only — use benchmark-competitiveness --dataset for AP",
        }

    def clear_cache(self) -> int:
        """Delete cached model weights. Returns bytes freed."""
        from visionservex.runtime.downloads import cache_clean

        if self._loaded:
            self.unload()
        return cache_clean(self.entry.id)

    @property
    def names(self) -> list[str]:
        """Class names for this model (COCO80 for detection models)."""
        from visionservex.core.normalizer import COCO80_NAMES

        if self.entry.task == "detect":
            return COCO80_NAMES
        return []

    def supports(self, operation: str) -> dict[str, Any]:
        """Check whether an operation is supported for this model.

        Returns a dict with keys: supported (bool), reason (str), hint (str).
        """
        return _capability_check(self.entry.id, self.entry, operation)

    def training_info(self) -> dict[str, Any]:
        """Return training and fine-tuning capabilities for this model."""
        return _training_capabilities(self.entry.id)

    def export_info(self) -> dict[str, Any]:
        """Return export capabilities for this model."""
        return _export_capabilities(self.entry.id)

    def capabilities(self) -> dict[str, Any]:
        """Return the canonical capability-truth object for this model (v3.15.0).

        One honest dict: legal status, engine/inference readiness, and the
        training/export truth. See :func:`model_capabilities`.
        """
        return model_capabilities(self.entry.id)

    # ------- task-specific public API (v3.17.0) -------
    # Thin, typed routers over predict(); each raises TaskNotSupportedError when
    # the model's registered task doesn't match (never silently mis-routes).

    def detect(
        self,
        image: Image.Image | bytes | str | Path,
        *,
        prompts: Sequence[str] | None = None,
        threshold: float | None = None,
        **kwargs: Any,
    ):
        """Object detection → ``DetectionResult`` / ``OpenVocabularyResult`` / OBB.

        Typed router over :meth:`predict` for the detection family
        (``detect`` / ``obb`` / ``open_vocab_detect``). Open-vocabulary detectors
        accept ``prompts=[...]``. Raises :class:`TaskNotSupportedError` when the
        model's registered task is not a detection task — never silently
        mis-routes (e.g. calling ``detect()`` on a classifier raises).
        """
        _require_task(self.entry, _DETECT_TASKS, "detect")
        return self.predict(image, prompts=prompts, threshold=threshold, **kwargs)

    def classify(self, image: Image.Image | bytes | str | Path, *, top_k: int = 5, **kwargs: Any):
        """Top-k image classification → ``ClassificationResult``."""
        _require_task(self.entry, _CLASSIFY_TASKS, "classify")
        return self.predict(image, top_k=top_k, **kwargs)

    def embed(self, image: Image.Image | bytes | str | Path, **kwargs: Any):
        """Image embedding / feature extraction → ``EmbeddingResult``."""
        _require_task(self.entry, _EMBED_TASKS, "embed")
        return self.predict(image, **kwargs)

    def segment(
        self,
        image: Image.Image | bytes | str | Path,
        *,
        boxes: list[list[float]] | None = None,
        points: list[list[float]] | None = None,
        point_labels: list[int] | None = None,
        prompts: Sequence[str] | None = None,
        **kwargs: Any,
    ):
        """Promptable / foundation / semantic segmentation → ``SegmentationResult``."""
        _require_task(self.entry, _SEGMENT_TASKS, "segment")
        return self.predict(
            image, boxes=boxes, points=points, point_labels=point_labels, prompts=prompts, **kwargs
        )

    def similarity(self, a: Any, b: Any) -> float:
        """Cosine similarity between two embeddings (``EmbeddingResult`` or arrays)."""
        import numpy as np

        from visionservex.runtime.embeddings import cosine_similarity

        va = getattr(a, "embedding", a)
        vb = getattr(b, "embedding", b)
        return float(
            cosine_similarity(
                np.asarray(va, dtype=np.float32).ravel(),
                np.asarray(vb, dtype=np.float32).ravel(),
            )
        )

    def correspond(
        self, source_image: Any, target_image: Any, *, source_region=None, **kwargs: Any
    ):
        """Semantic correspondence — provided by the dedicated INSID3 API, not here."""
        from visionservex.exceptions import TaskNotSupportedError

        raise TaskNotSupportedError(
            self.entry.id,
            "correspond",
            self.entry.task,
            hint=(
                "Semantic correspondence / in-context segmentation is the INSID3 API: "
                "visionservex.vsx.VSX.insid3(model_id).segment(query, ref, ref_mask)."
            ),
        )

    def train(self, dataset: str | Path, **kwargs: Any) -> dict[str, Any]:
        """Train / fine-tune this model on a dataset (engine-dependent).

        Supported for the LibreYOLO detector family (YOLOX / YOLOv9 / RT-DETR /
        D-FINE) via the permissive ``libreyolo`` package. Other families return
        a structured ``TRAINING_NOT_SUPPORTED`` dict rather than raising.

        Args:
            dataset: Path to a YOLO ``data.yaml`` or a directory containing one.
            **kwargs: Forwarded to the engine trainer (``epochs``, ``batch``,
                ``device``, ``imgsz``, ...). When ``device`` is omitted the
                engine auto-detects a GPU.

        Returns:
            The engine's normalized training-result dict, or a
            ``TRAINING_NOT_SUPPORTED`` envelope.
        """
        cap = _training_capabilities(self.entry.id)
        if not (cap.get("train_supported") or cap.get("finetune_supported")):
            return {
                "status": "TRAINING_NOT_SUPPORTED",
                "model_id": self.entry.id,
                "family": self.entry.family,
                "reason": cap.get("notes", "Training is not supported for this model."),
                "docs": cap.get("docs", ""),
            }
        train_fn = getattr(self.engine, "train", None)
        if train_fn is None:
            # The family IS trainable, but VisionServeX does not wrap its training
            # loop — e.g. RF-DETR trains via the mature `rfdetr` package's own API.
            # This is not a failure; it points to the native path and the reload
            # route VisionServeX does provide.
            return {
                "status": "TRAIN_VIA_NATIVE_API",
                "model_id": self.entry.id,
                "family": self.entry.family,
                "reason": cap.get(
                    "notes", "Train via this family's native package API, then reload here."
                ),
                "supported_dataset_formats": cap.get("supported_dataset_formats", []),
                "reload_hint": (
                    f"After training, reload for inference via "
                    f"VisionModel.from_checkpoint(ckpt, model_id='{self.entry.id}')."
                ),
                "docs": cap.get("docs", ""),
            }
        return train_fn(dataset, **kwargs)

    def val(
        self,
        data: str | None = None,
        *,
        dataset: str | None = None,
        max_images: int = 100,
        device: str | None = None,
        out: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate model AP on an annotated dataset.

        Args:
            data: Dataset path. Accepts ``yolo:<path>`` or ``coco-json:<img>:<ann>`` formats.
            dataset: Alias for ``data``.
            max_images: Maximum number of images to evaluate.
            device: Device override.
            out: Path prefix for output JSON/CSV.

        Returns:
            Evaluation result dict with ap50, map50_95, precision, recall, f1, latency.

        Notes:
            Only detection task is fully supported. Segmentation/pose/OBB
            return BENCHMARK_NOT_IMPLEMENTED.
        """
        if self.entry.task not in ("detect", "open_vocab_detect"):
            return {
                "status": "BENCHMARK_NOT_IMPLEMENTED",
                "task": self.entry.task,
                "message": f"AP evaluation for task={self.entry.task!r} is not yet implemented.",
                "hint": "Detection AP is available. Use visionservex benchmark benchmark-segmentation for roadmap.",
            }

        ds = data or dataset
        if not ds:
            return {
                "status": "DATASET_REQUIRED",
                "message": "Provide data= as 'yolo:<path>' or 'coco-json:<img_dir>:<ann_file>'",
                "hint": "visionservex val MODEL --dataset yolo:/path/to/coco128",
            }

        from pathlib import Path as _Path

        from visionservex.runtime.evaluation import (
            load_coco_json,
            load_yolo_format,
            run_model_on_dataset,
        )

        samples = None
        if ds.startswith("yolo:"):
            yolo_path = _Path(ds[5:])
            if not yolo_path.exists():
                return {
                    "status": "DATASET_NOT_FOUND",
                    "message": f"Path not found: {yolo_path}",
                    "hint": "Provide an existing directory with images/ and labels/ subdirs.",
                }
            samples, _ = load_yolo_format(yolo_path, max_images=max_images)
        elif ds.startswith("coco-json:"):
            parts = ds[10:].split(":", 1)
            if len(parts) == 2:
                images_p, ann_p = _Path(parts[0]), _Path(parts[1])
                if not images_p.exists() or not ann_p.exists():
                    return {
                        "status": "DATASET_NOT_FOUND",
                        "message": f"images_dir={images_p} or ann_file={ann_p} not found.",
                    }
                samples, _ = load_coco_json(images_p, ann_p, max_images=max_images)

        if samples is None:
            return {
                "status": "DATASET_PARSE_FAILED",
                "message": f"Could not load dataset from {ds!r}",
                "hint": "Use 'yolo:<path>' or 'coco-json:<img_dir>:<ann_file>'",
            }

        result = run_model_on_dataset(
            self.entry.id, samples, device=device or self.device, dataset_name=ds
        )

        if out:
            import json as _json

            op = _Path(out).with_suffix(".json")
            op.parent.mkdir(parents=True, exist_ok=True)
            op.write_text(_json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")

        return result.to_dict()

    # ------- helpers -------

    def _coerce_image(self, image: Image.Image | bytes | str | Path) -> Image.Image:
        limits = self.settings.limits
        if isinstance(image, Image.Image):
            w, h = image.size
            if w > limits.max_image_dim or h > limits.max_image_dim:
                raise ValueError(f"image dim {w}x{h} exceeds {limits.max_image_dim}")
            if w * h > limits.max_image_pixels:
                raise ValueError("image pixel area exceeds max_image_pixels")
            return image.convert("RGB") if image.mode != "RGB" else image
        if isinstance(image, (bytes, bytearray)):
            return open_safe(
                bytes(image), max_pixels=limits.max_image_pixels, max_dim=limits.max_image_dim
            )
        if isinstance(image, (str, Path)):
            return open_safe(
                Path(image), max_pixels=limits.max_image_pixels, max_dim=limits.max_image_dim
            )
        raise TypeError(f"unsupported image input type: {type(image).__name__}")


def _capability_check(model_id: str, entry: Any, operation: str) -> dict[str, Any]:
    """Return support status for an operation on a model."""
    op = operation.lower().replace("-", "_").replace(" ", "_")

    # Always supported for wired models
    always_ok = {"predict", "pull", "info", "benchmark", "cache_info", "checkpoint_info", "debug"}
    if op in always_ok:
        return {
            "supported": True,
            "reason": "core operation",
            "hint": f"visionservex {op} {model_id}",
        }

    # Export
    if op in ("export", "export_onnx"):
        from visionservex.core.model import _export_capabilities

        info = _export_capabilities(model_id)
        onnx_status = info.get("onnx", {}).get("status", "unsupported")
        return {
            "supported": onnx_status in ("supported", "experimental"),
            "status": onnx_status,
            "reason": info.get("onnx", {}).get("notes", ""),
            "hint": f"visionservex export {model_id} --format onnx",
        }

    # Training
    if op in ("train", "finetune", "fine_tune"):
        from visionservex.core.model import _training_capabilities

        info = _training_capabilities(model_id)
        sup = info.get("finetune_supported", False) or info.get("train_supported", False)
        return {
            "supported": sup,
            "reason": info.get("notes", "See training_info() for details"),
            "hint": f"visionservex model-card show {model_id} --json | jq '.visionservex_benchmark_status'",
        }

    # Tracking
    if op in ("track", "tracking"):
        family = getattr(entry, "family", "")
        supported = family in ("sam2",)
        return {
            "supported": supported,
            "reason": "SAM2 video tracking is experimental"
            if supported
            else "TRACKING_NOT_SUPPORTED",
            "hint": "visionservex track MODEL VIDEO"
            if supported
            else "Tracking not supported for this model.",
        }

    # Video
    if op in ("video", "video_predict"):
        return {
            "supported": False,
            "reason": "VIDEO_NOT_IMPLEMENTED",
            "hint": "Video inference is a roadmap item. Use batch_predict() on extracted frames.",
        }

    # Val/evaluate
    if op in ("val", "evaluate", "benchmark_ap"):
        task = getattr(entry, "task", "")
        sup = task in ("detect", "open_vocab_detect")
        return {
            "supported": sup,
            "reason": "AP50/mAP50:95 evaluation via benchmark-competitiveness"
            if sup
            else "BENCHMARK_NOT_IMPLEMENTED",
            "hint": f"visionservex val {model_id} --dataset yolo:/path/to/data"
            if sup
            else "See benchmark roadmap.",
        }

    return {
        "supported": False,
        "reason": f"Operation '{operation}' is not recognized.",
        "hint": "Use model.supports('predict'), model.supports('val'), model.supports('export'), etc.",
    }


# ---------------------------------------------------------------------------
# Static training capability tables
# ---------------------------------------------------------------------------

_TRAINING_TABLE: dict[str, dict[str, Any]] = {
    "dfine": {
        "train_supported": False,
        "finetune_supported": False,
        "resume_supported": False,
        "checkpoint_save_supported": False,
        "checkpoint_load_supported": False,
        "trained_checkpoint_predict_supported": False,
        "export_supported": ["onnx_experimental"],
        "supported_dataset_formats": [],
        "required_extra": "hf",
        "notes": (
            "TRAINING_NOT_SUPPORTED_IN_HF_BACKEND. HF Transformers does not expose "
            "D-FINE training. For a trainable, permissive D-FINE use the LibreYOLO "
            "variant `libreyolo-dfine-n` instead, or the official Peterande/D-FINE repo."
        ),
        "docs": "https://github.com/Peterande/D-FINE",
    },
    "rfdetr": {
        "train_supported": True,
        "finetune_supported": True,
        "resume_supported": True,
        "checkpoint_save_supported": True,
        "checkpoint_load_supported": True,
        "trained_checkpoint_predict_supported": True,
        "post_nms_predict_supported": True,
        "validated_lifecycle": True,
        "export_supported": ["onnx"],
        "supported_dataset_formats": ["coco-json", "yolo"],
        "required_extra": "rfdetr",
        "notes": (
            "RF-DETR trains/fine-tunes via the mature `rfdetr` package's native API "
            "(model.train(dataset_dir=...), COCO format). A trained checkpoint reloads "
            "for inference through VisionServeX via "
            "VisionModel.from_checkpoint(ckpt, model_id='rfdetr-nano') or "
            "engine.load_checkpoint(). Externally validated; Anastig-proven."
        ),
        "docs": "https://github.com/roboflow/rf-detr",
    },
    "libreyolo": {
        "train_supported": True,
        "finetune_supported": True,
        "resume_supported": True,
        "checkpoint_save_supported": True,
        "checkpoint_load_supported": True,
        "trained_checkpoint_predict_supported": True,
        "post_nms_predict_supported": True,
        "validated_lifecycle": True,
        "export_supported": ["onnx"],
        "supported_dataset_formats": ["yolo"],
        "required_extra": "libreyolo",
        "validated_variants": [
            "libreyolo-yolox-s",
            "libreyolo-yolov9-s",
            "libreyolo-rtdetr-r50",
        ],
        "known_blockers": ["libreyolo-dfine-*: UPSTREAM_DFINE_FDR_TOPK_CRASH (train blocked)"],
        "notes": (
            "LibreYOLO training via the permissive `libreyolo` package (YOLOX / "
            "YOLOv9 / RT-DETR). YOLO data.yaml dataset format; fine-tunes from "
            "COCO-pretrained base weights; saves best.pt/last.pt. v3.16.0 fixes the "
            "eval/=predict gap: EMA off by default (saves the actual trained weights, "
            "not the lagged EMA), predict uses the training imgsz, best.pt falls back "
            "to last.pt, and predict applies class-aware NMS. Validated live for "
            "yolox-s/yolov9-s/rtdetr-r50. D-FINE training is BLOCKED upstream "
            "(FDR topk crash) — inference-only. No Ultralytics. YOLO-NAS excluded."
        ),
        "docs": "https://github.com/LibreYOLO/libreyolo",
    },
    "torchvision-classify": {
        "train_supported": True,
        "finetune_supported": True,
        "resume_supported": False,
        "checkpoint_save_supported": True,
        "checkpoint_load_supported": True,
        "trained_checkpoint_predict_supported": True,
        "validated_lifecycle": True,
        "post_nms_predict_supported": True,
        "export_supported": ["onnx"],
        "supported_dataset_formats": ["imagefolder"],
        "required_extra": "torchvision",
        "validated_variants": ["torchvision-resnet18"],
        "known_blockers": [],
        "notes": (
            "Classic torchvision classifier fine-tune on an ImageFolder dataset "
            "(BSD-3-Clause). Saves best.pt/last.pt with class names; reload via "
            "VisionModel.from_checkpoint(model_id=...); ONNX export. Full lifecycle "
            "validated live for resnet18 (the loop is arch-generic). No Ultralytics."
        ),
        "docs": "https://github.com/pytorch/vision",
    },
    "swinv2": {
        "train_supported": False,
        "finetune_supported": False,
        "resume_supported": False,
        "checkpoint_save_supported": False,
        "checkpoint_load_supported": False,
        "export_supported": ["onnx_experimental"],
        "supported_dataset_formats": [],
        "required_extra": "hf",
        "notes": "TRAINING_NOT_SUPPORTED. SwinV2 classification fine-tuning is not wired in VisionServeX. Use HF Trainer or timm directly.",
        "docs": "https://github.com/microsoft/Swin-Transformer",
    },
    "sam": {
        "train_supported": False,
        "finetune_supported": False,
        "resume_supported": False,
        "checkpoint_save_supported": False,
        "checkpoint_load_supported": False,
        "export_supported": [],
        "supported_dataset_formats": [],
        "required_extra": "hf",
        "notes": "TRAINING_NOT_SUPPORTED. SAM/SAM2 training not exposed in this build.",
        "docs": "https://github.com/facebookresearch/segment-anything",
    },
    "sam2": {
        "train_supported": False,
        "finetune_supported": False,
        "resume_supported": False,
        "checkpoint_save_supported": False,
        "checkpoint_load_supported": False,
        "export_supported": [],
        "supported_dataset_formats": [],
        "required_extra": "hf",
        "notes": "TRAINING_NOT_SUPPORTED. SAM2 training not exposed in this build.",
        "docs": "https://github.com/facebookresearch/sam2",
    },
    "grounding-dino": {
        "train_supported": False,
        "finetune_supported": False,
        "resume_supported": False,
        "checkpoint_save_supported": False,
        "checkpoint_load_supported": False,
        "export_supported": [],
        "supported_dataset_formats": [],
        "required_extra": "hf",
        "notes": "TRAINING_NOT_SUPPORTED. Grounding DINO fine-tuning not wired.",
        "docs": "https://github.com/IDEA-Research/GroundingDINO",
    },
    "oneformer": {
        "train_supported": False,
        "finetune_supported": False,
        "resume_supported": False,
        "checkpoint_save_supported": False,
        "checkpoint_load_supported": False,
        "export_supported": [],
        "supported_dataset_formats": [],
        "required_extra": "hf",
        "notes": "TRAINING_NOT_SUPPORTED. OneFormer training not wired.",
        "docs": "https://github.com/SHI-Labs/OneFormer",
    },
    "default": {
        "train_supported": False,
        "finetune_supported": False,
        "resume_supported": False,
        "checkpoint_save_supported": False,
        "checkpoint_load_supported": False,
        "trained_checkpoint_predict_supported": False,
        "export_supported": [],
        "supported_dataset_formats": [],
        "notes": "TRAINING_NOT_SUPPORTED for this model family.",
    },
}

# ---------------------------------------------------------------------------
# Static export capability tables
# ---------------------------------------------------------------------------

_EXPORT_TABLE: dict[str, dict[str, Any]] = {
    "dfine": {
        "onnx": {
            "status": "experimental",
            "notes": "HF D-FINE ONNX export path not fully integrated. Use official repo script.",
        },
        "tensorrt": {"status": "unsupported", "notes": "TensorRT export not integrated."},
        "torchscript": {"status": "unsupported", "notes": "Not supported in HF backend."},
        "openvino": {"status": "unsupported", "notes": "Not integrated."},
        "hf_save_pretrained": {
            "status": "supported",
            "notes": "Use model.engine._model.save_pretrained(path) if loaded.",
        },
    },
    "rfdetr": {
        "onnx": {
            "status": "supported",
            "notes": "Use rfdetr package ONNX export. See rfdetr docs.",
        },
        "tensorrt": {
            "status": "backend_supported_but_not_integrated",
            "notes": "rfdetr supports TRT but not wired in VisionServeX.",
        },
        "torchscript": {"status": "unsupported", "notes": "Not supported."},
        "hf_save_pretrained": {"status": "unsupported", "notes": "Not HF-based."},
    },
    "libreyolo": {
        "onnx": {
            "status": "supported",
            "notes": "ONNX export via the libreyolo package exporter. "
            "VisionModel.export(format='onnx', output_path=...).",
        },
        "torchscript": {
            "status": "backend_supported_but_not_integrated",
            "notes": "libreyolo supports TorchScript; not surfaced/tested in VisionServeX v3.13.",
        },
        "tensorrt": {
            "status": "backend_supported_but_not_integrated",
            "notes": "libreyolo supports TensorRT; not surfaced/tested in VisionServeX v3.13.",
        },
        "openvino": {
            "status": "backend_supported_but_not_integrated",
            "notes": "libreyolo supports OpenVINO; not surfaced/tested in VisionServeX v3.13.",
        },
        "hf_save_pretrained": {"status": "unsupported", "notes": "Not HF-based."},
    },
    "torchvision-classify": {
        "onnx": {
            "status": "supported",
            "notes": "torch.onnx.export (opset 18, dynamic batch). "
            "VisionModel.export(format='onnx', output_path=...).",
        },
        "torchscript": {
            "status": "backend_supported_but_not_integrated",
            "notes": "torch.jit.script/trace available; not surfaced/tested in v3.15.",
        },
        "tensorrt": {"status": "unsupported", "notes": "Not integrated."},
        "hf_save_pretrained": {"status": "unsupported", "notes": "Not HF-based."},
    },
    "swinv2": {
        "onnx": {
            "status": "experimental",
            "notes": "HF transformers ONNX export may work via optimum.",
        },
        "tensorrt": {"status": "unsupported", "notes": "Not integrated."},
        "hf_save_pretrained": {
            "status": "supported",
            "notes": "SwinV2 is HF-based; save_pretrained() available.",
        },
    },
    "sam": {
        "onnx": {
            "status": "unsupported",
            "notes": "SAM ONNX export not integrated in VisionServeX.",
        },
        "tensorrt": {"status": "unsupported", "notes": "Not integrated."},
    },
    "sam2": {
        "onnx": {"status": "unsupported", "notes": "SAM2 ONNX export not integrated."},
        "tensorrt": {"status": "unsupported", "notes": "Not integrated."},
    },
    "grounding-dino": {
        "onnx": {"status": "unsupported", "notes": "Not integrated."},
        "hf_save_pretrained": {
            "status": "supported",
            "notes": "HF-based; save_pretrained() available.",
        },
    },
    "default": {
        "onnx": {"status": "unsupported", "notes": "Not supported for this model family."},
        "tensorrt": {"status": "unsupported", "notes": "Not integrated."},
    },
}


# LibreYOLO sub-families that are permissive (commercial-safe). YOLO-NAS shares
# family="libreyolo" but is Deci non-commercial and must never train.
_LIBREYOLO_TRAINABLE_SUBFAMILIES = frozenset({"yolox", "yolov9", "rtdetr", "dfine"})

# v3.16.0: variants whose FULL lifecycle (train -> checkpoint -> reload -> predict
# confident boxes -> NMS -> export) is live-validated. Only these report
# train-ready; larger/other variants are inference-ready (same engine, not
# individually validated) — we do NOT overclaim.
_LIBREYOLO_LIFECYCLE_VALIDATED = frozenset(
    {"libreyolo-yolox-s", "libreyolo-yolov9-s", "libreyolo-rtdetr-r50"}
)
_LIBREYOLO_DOCS = "https://github.com/LibreYOLO/libreyolo"


def _libreyolo_subfamily(model_id: str) -> str:
    """Extract the sub-family token from a ``libreyolo-<sub>-<size>`` id."""
    parts = model_id.split("-")
    return parts[1] if len(parts) >= 2 and parts[0] == "libreyolo" else ""


def _training_capabilities(model_id: str) -> dict[str, Any]:
    """Return training/fine-tuning capability dict for a model (per-variant truth)."""
    try:
        entry = default_registry().get(model_id)
        family = entry.family
    except Exception:
        family = model_id.split("-")[0]

    info = _TRAINING_TABLE.get(family, _TRAINING_TABLE["default"]).copy()
    info.setdefault("post_nms_predict_supported", False)
    info.setdefault("validated_lifecycle", False)
    info.setdefault("exact_blocker", None)

    # LibreYOLO training truth is PER SUB-FAMILY and PER VARIANT (v3.16.0).
    _sub = _libreyolo_subfamily(model_id)
    if family == "libreyolo" and _sub:
        not_supported = {
            **_TRAINING_TABLE["default"],
            "required_extra": "libreyolo",
            "trained_checkpoint_predict_supported": False,
            "validated_lifecycle": False,
            "docs": _LIBREYOLO_DOCS,
        }
        if _sub not in _LIBREYOLO_TRAINABLE_SUBFAMILIES:
            # YOLO-NAS (Deci non-commercial) and any non-permissive family.
            info = {
                **not_supported,
                "post_nms_predict_supported": False,
                "exact_blocker": "LIBREYOLO_NONCOMMERCIAL_FAMILY",
                "notes": (
                    "TRAINING_NOT_SUPPORTED: non-permissive LibreYOLO family. Only "
                    "YOLOX / YOLOv9 / RT-DETR train; YOLO-NAS is Deci non-commercial."
                ),
            }
        elif _sub == "dfine":
            # D-FINE training crashes upstream (libreyolo FDR 'selected index k out
            # of range'). Inference + reload of existing checkpoints still work;
            # only the train loop is blocked. (Validated in docs/qa/v316_*.)
            info = {
                **not_supported,
                "post_nms_predict_supported": True,
                "exact_blocker": "UPSTREAM_DFINE_FDR_TOPK_CRASH",
                "notes": (
                    "TRAINING_BLOCKED: libreyolo D-FINE training crashes upstream "
                    "(FDR topk 'selected index k out of range'). Inference is fully "
                    "supported; use libreyolo-yolox/yolov9/rtdetr to train."
                ),
            }
        elif model_id in _LIBREYOLO_LIFECYCLE_VALIDATED:
            info["trained_checkpoint_predict_supported"] = True
            info["post_nms_predict_supported"] = True
            info["validated_lifecycle"] = True
            info["exact_blocker"] = None
        else:
            # permissive yolox/yolov9/rtdetr variant NOT individually validated.
            info = {
                **not_supported,
                "post_nms_predict_supported": True,
                "exact_blocker": "VARIANT_NOT_LIFECYCLE_VALIDATED",
                "notes": (
                    "Inference-ready. Training uses the same validated libreyolo "
                    "engine, but this variant is not individually lifecycle-validated "
                    "in v3.16. Validated train targets: libreyolo-yolox-s / yolov9-s / "
                    "rtdetr-r50."
                ),
            }

    info["model_id"] = model_id
    info["family"] = family
    return info


def _export_capabilities(model_id: str) -> dict[str, Any]:
    """Return export capability dict for a model."""
    try:
        entry = default_registry().get(model_id)
        family = entry.family
    except Exception:
        family = model_id.split("-")[0]

    info = _EXPORT_TABLE.get(family, _EXPORT_TABLE["default"]).copy()
    info["model_id"] = model_id
    info["family"] = family
    return info


def _readiness_blocker(
    readiness_state: str, entry: Any, status: str, engine_registered: bool
) -> str | None:
    """A single canonical, human-readable blocker for a precise readiness state.

    Returns ``None`` for the live-ready states. For derived/blocked states it
    explains exactly why the model is not plug-and-play usable, reusing the
    registry's curated ``unavailable_reason`` where one exists.
    """
    from visionservex.readiness import taxonomy as _tx

    reason = entry.unavailable_reason
    table: dict[str, str] = {
        _tx.TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION: (
            "Train lifecycle is capability-derived; not live-verified in v3.18."
        ),
        _tx.INFERENCE_READY_DERIVED_NEEDS_LIVE_CONFIRMATION: (
            "Inference path is capability-derived; not live-verified in v3.18."
        ),
        _tx.GATED_TOKEN_REQUIRED: (
            "Gated: requires your own Hugging Face token and acceptance of the "
            "upstream license (BYOT). VisionServeX never ships the weights or token."
        ),
        _tx.LICENSE_BLOCKED: (
            "Copyleft/enterprise license is forbidden on the VisionServeX runtime/training path."
        ),
        _tx.NON_COMMERCIAL_BLOCKED: (
            "Non-commercial / research-only license; blocked from the commercial-safe default."
        ),
        _tx.CUSTOM_LOADER_REQUIRED: reason
        or "Custom loader required from the official upstream repository.",
        _tx.PARTIAL_IMPLEMENTATION_BLOCKED: reason
        or f"Partial implementation (implementation_status={status}).",
        _tx.CATALOG_ONLY_ENGINE_NOT_WIRED: reason
        or f"Catalog-only: engine {entry.engine!r} not wired (implementation_status={status}).",
        _tx.WEIGHTS_MISSING: reason or "Weights not released or unverifiable.",
        _tx.DEPENDENCY_MISSING: reason or "A required optional dependency is not installed.",
        _tx.UPSTREAM_CRASH: reason or "Upstream code crashes on this path.",
        _tx.OOM_BLOCKED: reason or "Out of memory for this device.",
        _tx.TASK_NOT_SUPPORTED: reason or "Task not supported by this engine.",
        _tx.UNKNOWN_REVIEW_REQUIRED: (
            "Unknown/custom license or unclassified state; hidden pending review."
        ),
    }
    if readiness_state in _tx.LIVE_READY_STATES:
        return None
    return table.get(readiness_state, reason)


def model_capabilities(model_id: str) -> dict[str, Any]:
    """Return the canonical capability-truth object for a model (v3.15.0).

    Assembles legal status (curated policy when present, else the registry
    license), engine/inference readiness, and the training/export truth into one
    honest dict. ``readiness`` is one of: ``train-ready`` (train + reloaded-
    checkpoint predict), ``inference-ready`` (wired engine + pretrained load),
    ``catalog-only`` (registry row but no wired runtime engine), or ``blocked``.
    """
    from visionservex.engines.registry import _FACTORIES
    from visionservex.licensing.policy import get_policy

    entry = default_registry().get(model_id)
    tcap = _training_capabilities(model_id)
    ecap = _export_capabilities(model_id)
    pol = get_policy(model_id)

    engine_registered = entry.engine in _FACTORIES
    status = entry.implementation_status
    inference_ready = status == "wired" and engine_registered
    train_ready = bool(
        tcap.get("train_supported") and tcap.get("trained_checkpoint_predict_supported")
    )

    from visionservex.readiness import live_evidence, taxonomy

    if pol is not None:
        legal_status = pol.final_policy
        gated = bool(pol.gated)
        license_code = pol.code_license
        license_weights = pol.weights_license
        policy_bucket: str | None = pol.final_policy
    else:
        # No curated policy row → fall back to the registry license, and DO NOT
        # claim commercial-safe-by-default (only the curated policy can grant that).
        legal_status = "registry_license_only"
        gated = bool(entry.requires_auth)
        license_code = entry.license
        license_weights = None
        policy_bucket = None

    # Canonical single license string + legal class. Governs the weights you
    # would actually run: weights license first, else code license, else registry.
    license_effective = license_weights or license_code or entry.license
    license_class = taxonomy.classify_license(license_effective)

    # commercial_safe is granted ONLY by a curated commercial_safe_core policy row,
    # and is hard-gated so a copyleft / non-commercial license can never be
    # commercial-safe even if a row were mis-entered (defense in depth).
    commercial_safe = bool(
        pol is not None
        and pol.default_safe
        and pol.final_policy == "commercial_safe_core"
        and license_class not in ("copyleft", "noncommercial")
    )
    requires_token = bool(gated)

    # Soft legal-review overlay: VisionServeX-flagged review, or a custom/unknown
    # license with no curated policy row → hidden from end users pending review.
    legal_review_required = bool(
        policy_bucket == "legal_review_required"
        or (pol is None and license_class in ("custom_unknown", "unknown"))
    )

    # Legacy coarse readiness — kept byte-identical for backward compatibility
    # (existing tests and the v3.17 matrix assert these four values + counts).
    if status == "stub" or not engine_registered:
        readiness = "catalog-only"
    elif train_ready and inference_ready:
        readiness = "train-ready"
    elif inference_ready:
        readiness = "inference-ready"
    else:
        readiness = "blocked"

    # v3.18 precise readiness state (canonical, machine-consumable) + visibility.
    live_inf = live_evidence.live_inference_verified(model_id)
    live_trn = live_evidence.live_train_verified(model_id)
    # v3.21: live via an isolated Docker sidecar (e.g. Florence-2). A legal/gated/
    # weights block still wins inside ``compute_readiness_state``.
    live_sidecar = live_evidence.live_sidecar_verified(model_id)
    readiness_state = taxonomy.compute_readiness_state(
        task=entry.task,
        implementation_status=status,
        engine=entry.engine,
        engine_registered=engine_registered,
        policy_bucket=policy_bucket,
        license_class=license_class,
        unavailable_reason=entry.unavailable_reason,
        train_ready=train_ready,
        inference_ready=inference_ready,
        live_inference_verified=live_inf,
        live_train_verified=live_trn,
        live_inference_blocker=live_evidence.live_inference_blocker(model_id),
        sidecar_live_verified=live_sidecar,
    )
    visibility = taxonomy.anastig_visibility(
        readiness_state,
        live_inference=live_inf,
        live_train=live_trn,
        task=entry.task,
        legal_review=legal_review_required,
    )
    blocker = _readiness_blocker(readiness_state, entry, status, engine_registered)

    # v3.21: a sidecar-live model is usable (via the sidecar), so its coarse legacy
    # bucket is "inference-ready" even though its host engine is an honest stub.
    # Without this the coarse view would contradict the precise sidecar state.
    if readiness_state in taxonomy.LIVE_SIDECAR_READY_STATES:
        readiness = "inference-ready"

    # v3.21 sidecar dimension. ``sidecar_required`` is True only when the model's
    # *sole* working path is the sidecar (i.e. it earned a ``*_READY_LIVE_SIDECAR``
    # state); a host-runnable model that merely *also* has a sidecar is not required
    # to use it. CPU was the verification device this sprint; GPU is unverified.
    sidecar_name = _SIDECAR_BY_FAMILY.get(entry.family)
    sidecar_supported = sidecar_name is not None
    sidecar_required = readiness_state in taxonomy.LIVE_SIDECAR_READY_STATES
    if sidecar_required:
        anastig_sidecar_visibility = visibility  # e.g. "show_inference_sidecar"
    elif sidecar_supported:
        anastig_sidecar_visibility = "host_preferred_sidecar_available"
    else:
        anastig_sidecar_visibility = "none"

    # v3.20: separate inference / train / fine-tune lifecycle dimensions, each with
    # a *_live_verified flag backed by the committed matrices.
    live_reload = live_evidence.live_reload_verified(model_id)
    live_export = live_evidence.live_export_verified(model_id)
    live_ftune = live_evidence.live_finetune_verified(model_id)
    reload_supported = bool(tcap.get("checkpoint_load_supported"))
    # A fine-tune path exists for trainable models and for embedding models (a
    # frozen-backbone head/linear-probe fine-tune). It is only *_live_verified once
    # the committed v3.20 train/finetune matrix proves it.
    sam_decoder_finetunable = entry.engine in _SAM_DECODER_FINETUNE_ENGINES
    fine_tune_ready = bool(
        train_ready
        or (entry.task in _EMBED_TASKS and inference_ready)
        or (entry.task in _SEGMENT_TASKS and sam_decoder_finetunable and inference_ready)
    )
    fine_tune_kind = _fine_tune_kind(
        train_ready=train_ready,
        task=entry.task,
        inference_ready=inference_ready,
        engine=entry.engine,
    )

    def _stage_visibility(supported: bool, live: bool, show_verb: str) -> str:
        if live:
            return show_verb
        # admin_only only when the engine is actually usable (inference-ready) but
        # the train/finetune lifecycle is not yet live-proven; a catalog-only model
        # whose engine is unwired cannot train, so it is hidden.
        if supported and inference_ready:
            return "admin_only"
        return "hide"

    anastig_train_visibility = _stage_visibility(train_ready, live_trn, "show_train")
    anastig_finetune_visibility = _stage_visibility(fine_tune_ready, live_ftune, "show_finetune")

    export_supported = [
        k for k, v in ecap.items() if isinstance(v, dict) and v.get("status") == "supported"
    ]

    exact_blocker: str | None = None
    if readiness == "catalog-only":
        exact_blocker = entry.unavailable_reason or (
            f"CATALOG_ONLY: engine {entry.engine!r} not wired (implementation_status={status})"
        )
    elif readiness == "blocked":
        exact_blocker = entry.unavailable_reason or (
            f"NOT_INFERENCE_READY: implementation_status={status}, engine_registered={engine_registered}"
        )
    elif not train_ready and tcap.get("exact_blocker"):
        # inference-ready detector whose TRAINING is blocked/not-validated: surface
        # the training blocker (e.g. UPSTREAM_DFINE_FDR_TOPK_CRASH).
        exact_blocker = tcap.get("exact_blocker")

    # v3.22.0 — TRUE-batch capability. Authoritative source is the engine class
    # (its ``supports_true_batch`` is what the forward-call test verifies), not
    # the static registry field. ``batch_path`` mirrors the /infer-batch contract.
    try:
        _eng = build_engine(entry)
        supports_true_batch = bool(getattr(_eng, "supports_true_batch", False))
        max_batch_size_hint = int(getattr(_eng, "max_batch_size_hint", 1))
        preferred_batch_sizes = list(getattr(_eng, "preferred_batch_sizes", (1,)))
    except Exception:
        supports_true_batch = False
        max_batch_size_hint = 1
        preferred_batch_sizes = [1]
    batch_path = (
        "true_tensor_batch"
        if supports_true_batch
        else ("internal_loop" if engine_registered else "unsupported")
    )

    return {
        "supports_true_batch": supports_true_batch,
        "batch_path": batch_path,
        "max_batch_size_hint": max_batch_size_hint,
        "preferred_batch_sizes": preferred_batch_sizes,
        "registry_batch_support_claim": bool(entry.batch_support),
        "model_id": model_id,
        "family": entry.family,
        "task": entry.task,
        "engine": entry.engine,
        "backend": entry.backend,
        "model_category": str(entry.model_category) if entry.model_category else None,
        "readiness": readiness,
        "readiness_state": readiness_state,
        "anastig_visibility": visibility,
        "blocker": blocker,
        "legal_status": legal_status,
        "license": license_effective,
        "license_class": license_class,
        "commercial_safe": commercial_safe,
        "gated": gated,
        "requires_token": requires_token,
        "legal_review_required": legal_review_required,
        "license_code": license_code,
        "license_weights": license_weights,
        "license_registry": entry.license,
        "has_policy_row": pol is not None,
        "implementation_status": status,
        "engine_registered": engine_registered,
        "predict_supported": inference_ready,
        "live_verified_inference": live_inf,
        "live_verified_train": live_trn,
        # v3.20: explicit per-lifecycle-dimension truth for Anastig.
        "inference_ready": inference_ready,
        "inference_live_verified": live_inf,
        "train_ready": train_ready,
        "train_live_verified": live_trn,
        "fine_tune_ready": fine_tune_ready,
        "fine_tune_live_verified": live_ftune,
        "fine_tune_kind": fine_tune_kind,
        "reload_supported": reload_supported,
        "reload_live_verified": live_reload,
        "export_live_verified": live_export,
        "token_never_logged": True,
        "anastig_train_visibility": anastig_train_visibility,
        "anastig_finetune_visibility": anastig_finetune_visibility,
        # v3.21: isolated-sidecar dimension (Florence-2, OpenMMLab, DEIMv2, ...).
        "sidecar_supported": sidecar_supported,
        "sidecar_required": sidecar_required,
        "sidecar_name": sidecar_name,
        "sidecar_live": live_sidecar,
        "sidecar_cpu_verified": live_sidecar,
        "sidecar_gpu_verified": False,
        "anastig_sidecar_visibility": anastig_sidecar_visibility,
        "pretrained_inference_supported": inference_ready,
        "pretrained_load_supported": inference_ready and entry.download_type != "not_available",
        "auto_download": bool(entry.auto_download),
        "required_extra": entry.install_extra,
        "train_supported": bool(tcap.get("train_supported")),
        "finetune_supported": bool(tcap.get("finetune_supported")),
        "checkpoint_save_supported": bool(tcap.get("checkpoint_save_supported")),
        "checkpoint_load_supported": bool(tcap.get("checkpoint_load_supported")),
        "trained_checkpoint_predict_supported": bool(
            tcap.get("trained_checkpoint_predict_supported")
        ),
        "post_nms_predict_supported": bool(tcap.get("post_nms_predict_supported")),
        "validated_lifecycle": bool(tcap.get("validated_lifecycle")),
        "export_supported": export_supported,
        "supported_dataset_formats": tcap.get("supported_dataset_formats", []),
        "validated_variants": tcap.get("validated_variants", []),
        "known_blockers": tcap.get("known_blockers", []),
        "exact_blocker": exact_blocker,
        "tasks": [entry.task],
        "validated_syntax": _validated_syntax(model_id, entry.task, tcap, export_supported),
    }


def _validated_syntax(
    model_id: str, task: str, tcap: dict[str, Any], export_supported: list[str]
) -> dict[str, str]:
    """The public-API call syntax that applies to this model (v3.17.0)."""
    syn = {"predict": f'VisionModel("{model_id}").predict("image.jpg")'}
    if task in _CLASSIFY_TASKS:
        syn["classify"] = f'VisionModel("{model_id}").classify("image.jpg", top_k=5)'
    if task in _EMBED_TASKS:
        syn["embed"] = f'VisionModel("{model_id}").embed("image.jpg")'
        syn["similarity"] = 'model.similarity(model.embed("a.jpg"), model.embed("b.jpg"))'
    if task in _SEGMENT_TASKS:
        syn["segment"] = f'VisionModel("{model_id}").segment("image.jpg", boxes=[[x, y, w, h]])'
    if task in _DETECT_TASKS:
        syn["detect"] = f'VisionModel("{model_id}").detect("image.jpg", threshold=0.25)'
    if task in _OPEN_VOCAB_TASKS:
        syn["detect"] = f'VisionModel("{model_id}").detect("image.jpg", prompts=["cat", "dog"])'
    if tcap.get("train_supported"):
        syn["train"] = f'VisionModel("{model_id}").train("data.yaml", epochs=1)'
        syn["from_checkpoint"] = f'VisionModel.from_checkpoint(ckpt, model_id="{model_id}")'
    if export_supported:
        syn["export"] = f'VisionModel("{model_id}").export(format="onnx", output_path="m.onnx")'
    return syn


__all__ = ["VisionModel", "list_models", "model_capabilities"]
