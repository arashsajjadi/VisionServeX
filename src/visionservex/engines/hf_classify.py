# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Generic HF classification engine.

Supports any model family that HF AutoModelForImageClassification covers:
ConvNeXtV2, EfficientNet, ViT, DeiT, BEiT, MaxViT, Swin, etc.

Uses ``AutoModelForImageClassification`` + ``AutoImageProcessor`` from
Hugging Face Transformers, extracting top-k logits and returning a
``ClassificationResult``.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from visionservex.core.results import BaseResult, ClassificationResult
from visionservex.engines._stub import StubEngine, assert_modules
from visionservex.engines.base import MissingDependencyError
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)


class HFClassifyEngine(StubEngine):
    """Generic HF image-classification engine (AutoModelForImageClassification)."""

    real_install_extra = "hf"
    real_modules = ("transformers", "torch")
    backend_label = "huggingface_classify"

    def __init__(self, entry: ModelEntry) -> None:
        super().__init__(entry)
        self._processor: Any = None
        self._model: Any = None
        self._id2label: dict[int, str] = {}
        self._torch: Any = None

    def _real_load(self, *, device: str, precision: str) -> None:
        assert_modules(self.real_modules, install_hint=self._install_hint())
        if not self.entry.hf_repo_id:
            raise MissingDependencyError(
                f"model {self.entry.id!r} has no hf_repo_id",
                install_hint=self._install_hint(),
            )

        # timm/ HF repos require the `timm` package and a timm-aware loader.
        # AutoModelForImageClassification does NOT load timm/ repos via the
        # standard path; attempting it raises a KeyError or class_not_found.
        # We surface this as an actionable expected_blocker.
        if (self.entry.hf_repo_id or "").startswith("timm/"):
            try:
                import timm as _timm  # type: ignore  # noqa: F401
            except ImportError as exc:
                raise MissingDependencyError(
                    f"TIMM_REQUIRED: model {self.entry.id!r} uses a timm/ HF "
                    f"repo ({self.entry.hf_repo_id}) which requires the `timm` "
                    f"package. Install: pip install timm",
                    install_hint="pip install timm",
                ) from exc

        from visionservex.runtime.downloads import download

        download(self.entry)

        import torch  # type: ignore
        from transformers import AutoImageProcessor, AutoModelForImageClassification  # type: ignore

        torch_dtype = torch.float32
        if precision == "fp16" and device != "cpu":
            torch_dtype = torch.float16
        elif precision == "bf16" and device != "cpu":
            torch_dtype = torch.bfloat16

        kwargs: dict[str, Any] = {}
        if torch_dtype is not torch.float32:
            kwargs["torch_dtype"] = torch_dtype

        self._processor = AutoImageProcessor.from_pretrained(self.entry.hf_repo_id)
        self._model = AutoModelForImageClassification.from_pretrained(
            self.entry.hf_repo_id, **kwargs
        )
        self._model.to(device)
        self._model.eval()
        self._id2label = getattr(self._model.config, "id2label", {})
        self._torch = torch

    def unload(self) -> None:
        import contextlib

        if self._model is not None:
            with contextlib.suppress(Exception):
                del self._model, self._processor
            self._model = None
            self._processor = None
        super().unload()

    def predict(
        self,
        image: Image.Image,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> BaseResult:
        if not self._real_ready:
            return super().predict(image, **kwargs)

        inputs = self._processor(images=image, return_tensors="pt")
        model_device = next(self._model.parameters()).device
        model_dtype = next(self._model.parameters()).dtype
        inputs_dev = {}
        for k, v in inputs.items():
            v = v.to(device=model_device)
            if v.is_floating_point():
                v = v.to(dtype=model_dtype)
            inputs_dev[k] = v

        with self._torch.no_grad():
            out = self._model(**inputs_dev)

        logits = out.logits[0]
        probs = self._torch.nn.functional.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(min(top_k, len(probs)))

        top_k_pairs = [
            (self._id2label.get(int(i), str(int(i))), float(p))
            for i, p in zip(topk_idx.cpu(), topk_probs.cpu(), strict=False)
        ]

        return ClassificationResult(
            kind="classification",
            model_id=self.entry.id,
            task="classify",
            image_size=image.size,
            device=self.device,
            top_k=top_k_pairs,
            metadata={"backend": self.backend_label, "precision": self.precision},
        )


def _factory(entry: ModelEntry) -> HFClassifyEngine:
    return HFClassifyEngine(entry)


for _name in ("hf_classify", "convnextv2", "maxvit", "efficientnet", "vit", "deit", "beit"):
    register_engine(_name, _factory)

__all__ = ["HFClassifyEngine"]
