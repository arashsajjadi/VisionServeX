# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Swin Transformer V2 classification engine via HF Transformers.

Uses ``AutoModelForImageClassification`` and ``AutoImageProcessor`` from
the Hugging Face ``transformers`` library. Outputs top-k class predictions
for any SwinV2 (or other ImageNet-trained) HF model.

Install:
    pip install 'visionservex[hf]'

Supported model IDs:
    swinv2-tiny, swinv2-small, swinv2-base, swinv2-large
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from visionservex.core.results import BaseResult, ClassificationResult
from visionservex.engines._stub import StubEngine
from visionservex.engines.base import MissingDependencyError
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)


class SwinV2Engine(StubEngine):
    """Real HF-backed classification engine for SwinV2 and compatible models."""

    real_install_extra = "hf"
    real_modules = ("transformers", "torch")
    backend_label = "huggingface"

    def __init__(self, entry: ModelEntry) -> None:
        super().__init__(entry)
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._id2label: dict[int, str] = {}

    # ------ lifecycle ------

    def _real_load(self, *, device: str, precision: str) -> None:
        if not self.entry.hf_repo_id:
            raise MissingDependencyError(
                f"model {self.entry.id!r} has no hf_repo_id",
                install_hint=self._install_hint(),
            )
        # Trigger download first
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

        _log.info("loading %s from %s", self.entry.id, self.entry.hf_repo_id)
        self._processor = AutoImageProcessor.from_pretrained(self.entry.hf_repo_id, use_fast=True)
        self._model = AutoModelForImageClassification.from_pretrained(
            self.entry.hf_repo_id, **kwargs
        )
        self._model.to(device)
        self._model.eval()
        self._torch = torch
        # Build id2label mapping
        cfg = self._model.config
        self._id2label = getattr(cfg, "id2label", {}) or {}
        _log.info("%s loaded with %d classes", self.entry.id, len(self._id2label))

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
        super().unload()

    def warmup(self) -> None:
        if not self._real_ready:
            return
        try:
            dummy = Image.new("RGB", (256, 256), "gray")
            self.predict(dummy)
        except Exception:
            pass

    # ------ inference ------

    def predict(
        self,
        image: Image.Image,
        *,
        prompts: Sequence[str] | None = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> BaseResult:
        if not self._real_ready:
            return super().predict(image, prompts=prompts, **kwargs)

        model_device = next(self._model.parameters()).device
        model_dtype = next(self._model.parameters()).dtype
        inputs = self._processor(images=image, return_tensors="pt")
        # Cast inputs appropriately
        inputs_dev: dict[str, Any] = {}
        for k, v in inputs.items():
            v = v.to(device=model_device)
            if v.is_floating_point():
                v = v.to(dtype=model_dtype)
            inputs_dev[k] = v

        with self._torch.no_grad():
            out = self._model(**inputs_dev)

        logits = out.logits[0]
        probs = self._torch.softmax(logits.float(), dim=-1)
        top = self._torch.topk(probs, min(top_k, len(probs)))
        results = []
        for idx, score in zip(top.indices.tolist(), top.values.tolist(), strict=False):
            label = self._id2label.get(idx, f"class_{idx}")
            results.append((label, float(score)))

        return ClassificationResult(
            kind="classification",
            model_id=self.entry.id,
            task=self.entry.task,
            image_size=image.size,
            device=self.device,
            precision=self.precision,
            backend=self.backend_label,
            top_k=results,
        )

    def infer(self, preprocessed: Any, **kwargs: Any) -> Any:
        return preprocessed

    def postprocess(self, raw: Any, *, image: Any, **kwargs: Any) -> BaseResult:
        return self._mock.postprocess(raw, image=image, **kwargs)

    def export(self, format: str, output_path) -> Path:

        out = Path(output_path)
        if format.lower() not in {"onnx"}:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support export to {format!r}. "
                "Only 'onnx' is supported for SwinV2."
            )
        if not self._real_ready:
            raise RuntimeError("model must be loaded before export. Call load() first.")
        out.parent.mkdir(parents=True, exist_ok=True)
        import torch  # type: ignore
        from PIL import Image as _Image

        proc = self._processor
        dummy_img = _Image.new("RGB", (256, 256))
        inputs = proc(images=dummy_img, return_tensors="pt")
        dummy = inputs["pixel_values"]

        self._model.cpu()
        self._model.eval()
        torch.onnx.export(
            self._model.cpu(),
            (dummy,),
            str(out),
            opset_version=17,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch_size"}},
        )
        # Move model back if needed
        if self.device != "cpu":
            self._model.to(self.device)
        return out


def _factory(entry: ModelEntry) -> SwinV2Engine:
    return SwinV2Engine(entry)


register_engine("swinv2", _factory)

__all__ = ["SwinV2Engine"]
