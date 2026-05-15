# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""D-FINE object detection engine via Hugging Face Transformers.

Uses ``AutoModelForObjectDetection`` and ``AutoImageProcessor`` from the
``ustc-community`` D-FINE checkpoints on Hugging Face.

Supported model IDs → HF repo mapping:
    dfine-n  → ustc-community/dfine-nano-coco
    dfine-s  → ustc-community/dfine-small-obj2coco  (default)
    dfine-m  → ustc-community/dfine-medium-obj2coco
    dfine-l  → ustc-community/dfine-large-obj2coco-e25
    dfine-x  → ustc-community/dfine-xlarge-obj2coco

Install:
    pip install 'visionservex[dfine]'   (or [hf])
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image

from visionservex.core.results import BaseResult, Box, Detection, DetectionResult
from visionservex.engines._stub import StubEngine
from visionservex.engines.base import MissingDependencyError
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)

# Maps VisionServeX model id → HF repo id
_HF_REPOS: dict[str, str] = {
    "dfine-n": "ustc-community/dfine-nano-coco",
    "dfine-s": "ustc-community/dfine-small-obj2coco",
    "dfine-m": "ustc-community/dfine-medium-obj2coco",
    "dfine-l": "ustc-community/dfine-large-obj2coco-e25",
    "dfine-x": "ustc-community/dfine-xlarge-obj2coco",
}


def _resolve_repo(entry: ModelEntry) -> str:
    """Return the HF repo for a registry entry, preferring hf_repo_id."""
    if entry.hf_repo_id:
        return entry.hf_repo_id
    repo = _HF_REPOS.get(entry.id)
    if repo is None:
        raise MissingDependencyError(
            f"no HF repo mapped for D-FINE model id {entry.id!r}",
            install_hint="check `visionservex list-models --family dfine`",
        )
    return repo


class DFINEEngine(StubEngine):
    """Real D-FINE detection engine backed by HF Transformers (``d_fine`` model type)."""

    real_install_extra = "dfine"
    real_modules = ("transformers", "torch")
    backend_label = "huggingface_dfine"

    def __init__(self, entry: ModelEntry) -> None:
        super().__init__(entry)
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._id2label: dict[int, str] = {}

    # ------ lifecycle ------

    def _real_load(self, *, device: str, precision: str) -> None:
        from visionservex.runtime.downloads import download

        # Temporarily patch entry's hf_repo_id so the downloader targets the right repo.
        repo = _resolve_repo(self.entry)
        if not self.entry.hf_repo_id:
            self.entry.hf_repo_id = repo  # type: ignore[misc]

        download(self.entry)

        import torch  # type: ignore
        from transformers import AutoImageProcessor, AutoModelForObjectDetection  # type: ignore

        torch_dtype = torch.float32
        if precision in ("fp16", "bf16") and device != "cpu":
            torch_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

        # Use `dtype` (new API); fall back to `torch_dtype` for older transformers.
        kwargs: dict[str, Any] = {}
        if torch_dtype is not torch.float32:
            kwargs["dtype"] = torch_dtype  # new name in transformers ≥ 4.x

        _log.info("loading D-FINE %s from %s on %s", self.entry.id, repo, device)
        self._processor = AutoImageProcessor.from_pretrained(repo, use_fast=True)
        try:
            self._model = AutoModelForObjectDetection.from_pretrained(repo, **kwargs)
        except TypeError:
            # Older transformers expects `torch_dtype`
            renamed = {("torch_dtype" if k == "dtype" else k): v for k, v in kwargs.items()}
            self._model = AutoModelForObjectDetection.from_pretrained(repo, **renamed)
        self._model.to(device)
        self._model.eval()
        self._torch = torch

        cfg = self._model.config
        self._id2label = {int(k): v for k, v in (getattr(cfg, "id2label", {}) or {}).items()}
        _log.info("D-FINE %s ready (%d classes)", self.entry.id, len(self._id2label))

    def unload(self) -> None:
        if self._model is not None:
            del self._model, self._processor
            self._model = None
            self._processor = None
        super().unload()

    def warmup(self) -> None:
        if not self._real_ready:
            return
        try:
            dummy = Image.new("RGB", (64, 64), "black")
            self.predict(dummy)
        except Exception:
            pass

    # ------ inference ------

    def predict(
        self,
        image: Image.Image,
        *,
        prompts: Sequence[str] | None = None,
        threshold: float = 0.3,
        **kwargs: Any,
    ) -> BaseResult:
        if not self._real_ready:
            return super().predict(image, prompts=prompts, **kwargs)

        model_device = next(self._model.parameters()).device
        model_dtype = next(self._model.parameters()).dtype
        w, h = image.size

        inputs = self._processor(images=[image], return_tensors="pt")
        inputs_dev: dict[str, Any] = {}
        for k, v in inputs.items():
            v = v.to(device=model_device)
            if v.is_floating_point():
                v = v.to(dtype=model_dtype)
            inputs_dev[k] = v

        with self._torch.no_grad():
            out = self._model(**inputs_dev)

        results = self._processor.post_process_object_detection(
            out, threshold=threshold, target_sizes=[(h, w)]
        )
        first = results[0]

        detections: list[Detection] = []
        for box, score, label_id in zip(
            first["boxes"].tolist(),
            first["scores"].tolist(),
            first["labels"].tolist(),
            strict=False,
        ):
            label = self._id2label.get(int(label_id), f"class_{label_id}")
            detections.append(
                Detection(
                    box=Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                    score=float(score),
                    label=label,
                    class_id=int(label_id),
                )
            )

        return DetectionResult(
            kind="detection",
            model_id=self.entry.id,
            task=self.entry.task,
            image_size=(w, h),
            device=self.device,
            precision=self.precision,
            backend=self.backend_label,
            detections=detections,
        )

    def infer(self, preprocessed: Any, **kwargs: Any) -> Any:
        return preprocessed

    def postprocess(self, raw: Any, *, image: Any, **kwargs: Any) -> BaseResult:
        return self._mock.postprocess(raw, image=image, **kwargs)


def _factory(entry: ModelEntry) -> DFINEEngine:
    return DFINEEngine(entry)


register_engine("dfine", _factory)

__all__ = ["DFINEEngine"]
