# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""OWLv2 — open-vocabulary detection engine.

Real implementation via Hugging Face Transformers:
``Owlv2Processor`` + ``Owlv2ForObjectDetection``.

Reference:
- https://huggingface.co/docs/transformers/en/model_doc/owlv2
- https://huggingface.co/google/owlv2-base-patch16-ensemble
- https://huggingface.co/google/owlv2-large-patch14-ensemble

Inputs:
- ``image``: a PIL Image.
- ``prompts``: a list of free-form text queries (e.g. ``["person", "red shirt", "car"]``).

The model assigns each detection to one of the supplied prompt strings.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image

from visionservex.core.results import (
    BaseResult,
    Box,
    Detection,
    OpenVocabularyResult,
)
from visionservex.engines._stub import StubEngine, assert_modules
from visionservex.engines.base import MissingDependencyError
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)


class OWLv2Engine(StubEngine):
    """OWLv2 zero-shot detection engine."""

    real_install_extra = "hf"
    real_modules = ("transformers", "torch")
    backend_label = "huggingface_owlv2"

    def __init__(self, entry: ModelEntry) -> None:
        super().__init__(entry)
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None

    # ----- lifecycle -----

    def _real_load(self, *, device: str, precision: str) -> None:
        assert_modules(self.real_modules, install_hint=self._install_hint())
        if not self.entry.hf_repo_id:
            raise MissingDependencyError(
                f"model {self.entry.id!r} has no hf_repo_id; cannot load from Hugging Face",
                install_hint=self._install_hint(),
            )

        from visionservex.runtime.downloads import download

        download(self.entry)

        import torch  # type: ignore

        torch_dtype = torch.float32
        if precision == "fp16" and device != "cpu":
            torch_dtype = torch.float16
        elif precision == "bf16" and device != "cpu":
            torch_dtype = torch.bfloat16

        kwargs: dict[str, Any] = {}
        if torch_dtype is not torch.float32:
            kwargs["torch_dtype"] = torch_dtype

        # OWL-ViT and OWLv2 use different HF classes; detect from family.
        _is_owlvit = self.entry.family.lower() in {"owlvit", "owl-vit", "owl_vit"}
        if _is_owlvit:
            from transformers import OwlViTForObjectDetection, OwlViTProcessor  # type: ignore

            self._processor = OwlViTProcessor.from_pretrained(self.entry.hf_repo_id)
            self._model = OwlViTForObjectDetection.from_pretrained(self.entry.hf_repo_id, **kwargs)
        else:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor  # type: ignore

            self._processor = Owlv2Processor.from_pretrained(self.entry.hf_repo_id)
            self._model = Owlv2ForObjectDetection.from_pretrained(self.entry.hf_repo_id, **kwargs)
        self._model.to(device)
        self._model.eval()
        self._torch = torch

    def warmup(self) -> None:
        if not self._real_ready:
            return
        import contextlib

        with contextlib.suppress(Exception):
            self.predict(Image.new("RGB", (64, 64), color="black"), prompts=["object"])

    def unload(self) -> None:
        if self._real_ready and self._model is not None:
            try:
                del self._model
                del self._processor
            except Exception:
                pass
            self._model = None
            self._processor = None
            self._real_ready = False
        super().unload()

    # ----- inference -----

    def predict(
        self,
        image: Image.Image,
        *,
        prompts: Sequence[str] | None = None,
        prompt: str | None = None,
        threshold: float = 0.1,
        **kwargs: Any,
    ) -> BaseResult:
        if not self._real_ready:
            return super().predict(image, prompts=prompts, prompt=prompt, **kwargs)

        # Accept either prompts (list) or prompt (comma-separated string).
        queries: list[str] = []
        if prompts:
            queries = [str(p).strip() for p in prompts if str(p).strip()]
        elif prompt:
            queries = [p.strip() for p in str(prompt).split(",") if p.strip()]
        if not queries:
            queries = ["object"]

        # OWLv2 expects text as nested list: [["query1", "query2", ...]]
        inputs = self._processor(text=[queries], images=image, return_tensors="pt")
        model_device = next(self._model.parameters()).device
        model_dtype = next(self._model.parameters()).dtype
        inputs_on_device = {}
        for k, v in inputs.items():
            v = v.to(device=model_device)
            if v.is_floating_point():
                v = v.to(dtype=model_dtype)
            inputs_on_device[k] = v

        with self._torch.no_grad():
            outputs = self._model(**inputs_on_device)

        target_sizes = self._torch.tensor([image.size[::-1]])  # (H, W)
        # The fast Owlv2Processor routes post-processing through its image_processor
        # sub-component (post_process_object_detection lives there, not on the wrapper).
        # We also accept the legacy path where the processor exposes it directly.
        _post_proc = getattr(
            self._processor,
            "post_process_object_detection",
            None,
        ) or getattr(
            getattr(self._processor, "image_processor", None),
            "post_process_object_detection",
            None,
        )
        if _post_proc is None:
            raise AttributeError(
                "Could not locate post_process_object_detection on "
                f"{type(self._processor).__name__}. "
                "This may indicate an unsupported transformers version."
            )
        results = _post_proc(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold,
        )
        first = results[0] if results else {"boxes": [], "scores": [], "labels": []}

        detections: list[Detection] = []
        for box, score, label_idx in zip(
            first["boxes"], first["scores"], first["labels"], strict=False
        ):
            xyxy = [float(v) for v in (box.tolist() if hasattr(box, "tolist") else box)]
            try:
                label_int = int(label_idx.item() if hasattr(label_idx, "item") else label_idx)
            except Exception:
                label_int = 0
            label_str = queries[label_int] if 0 <= label_int < len(queries) else str(label_int)
            detections.append(
                Detection(
                    box=Box(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]),
                    score=float(score),
                    label=label_str,
                    class_id=label_int,
                )
            )

        return OpenVocabularyResult(
            kind="open_vocab",
            model_id=self.entry.id,
            task=self.entry.task,
            image_size=image.size,
            device=self.device,
            prompts=queries,
            detections=detections,
            metadata={
                "backend": self.backend_label,
                "precision": self.precision,
                "threshold": threshold,
            },
        )


def _factory(entry: ModelEntry) -> OWLv2Engine:
    return OWLv2Engine(entry)


register_engine("owlv2", _factory)
register_engine("owlvit", _factory)  # OWL-ViT v1 uses same engine; family determines HF class

__all__ = ["OWLv2Engine"]
