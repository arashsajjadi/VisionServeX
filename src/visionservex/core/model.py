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
            return  # engine.load() will surface a clearer error if it needs weights
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

        self._ensure_loaded()
        pil = self._coerce_image(image)
        start = time.perf_counter()
        result = self.engine.predict(pil, prompts=prompts, **kwargs)
        result.latency_ms = (time.perf_counter() - start) * 1000.0
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
        if unload_after:
            self.unload()
        return result

    def batch_predict(
        self,
        images: Iterable[Image.Image | bytes | str | Path],
        *,
        prompts: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[BaseResult]:
        results: list[BaseResult] = []
        for img in images:
            results.append(self.predict(img, prompts=prompts, **kwargs))
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
    def from_checkpoint(cls, checkpoint_path: str | Path, **kwargs: Any) -> VisionModel:
        """Not generally supported. Raises a structured error with a hint."""
        raise NotImplementedError(
            "CHECKPOINT_LOAD_UNSUPPORTED: VisionModel does not support loading arbitrary "
            "local checkpoints. Use VisionModel('MODEL_ID').pull() to download an official "
            "checkpoint, or register a custom backend with visionservex model register-custom."
        )

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
        "export_supported": ["onnx_experimental"],
        "supported_dataset_formats": [],
        "required_extra": "hf",
        "notes": "TRAINING_NOT_SUPPORTED_IN_HF_BACKEND. HF Transformers does not expose D-FINE training. Use official Peterande/D-FINE repository for training.",
        "docs": "https://github.com/Peterande/D-FINE",
    },
    "rfdetr": {
        "train_supported": True,
        "finetune_supported": True,
        "resume_supported": True,
        "checkpoint_save_supported": True,
        "checkpoint_load_supported": True,
        "export_supported": ["onnx"],
        "supported_dataset_formats": ["coco-json", "yolo"],
        "required_extra": "rfdetr",
        "notes": "RF-DETR supports training/fine-tuning via the rfdetr package. Use rfdetr.train() directly.",
        "docs": "https://github.com/roboflow/rf-detr",
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


def _training_capabilities(model_id: str) -> dict[str, Any]:
    """Return training/fine-tuning capability dict for a model."""
    try:
        entry = default_registry().get(model_id)
        family = entry.family
    except Exception:
        family = model_id.split("-")[0]

    info = _TRAINING_TABLE.get(family, _TRAINING_TABLE["default"]).copy()
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


__all__ = ["VisionModel"]
