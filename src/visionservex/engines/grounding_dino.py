# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Grounding DINO engine.

Real implementation via Hugging Face Transformers:
``AutoProcessor`` + ``AutoModelForZeroShotObjectDetection``.

The model card name comes from the registry entry's ``hf_repo_id``. When
``transformers`` / ``torch`` are not installed the engine raises a friendly
:class:`MissingDependencyError`. Mock fallback only happens when the user
sets ``VISIONSERVEX_MODELS__ALLOW_MOCK_FALLBACK=true``.
"""

from __future__ import annotations

from typing import Any, Sequence

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


class GroundingDINOEngine(StubEngine):
    """Real HF-backed Grounding DINO engine with safe-stub semantics."""

    real_install_extra = "grounding"
    real_modules = ("transformers", "torch")
    backend_label = "huggingface"

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
        # Make sure weights are present locally; let the downloader handle progress.
        from visionservex.runtime.downloads import download

        download(self.entry)

        import torch  # type: ignore
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # type: ignore

        torch_dtype = torch.float32
        if precision == "fp16" and device != "cpu":
            torch_dtype = torch.float16
        elif precision == "bf16" and device != "cpu":
            torch_dtype = torch.bfloat16

        kwargs: dict[str, Any] = {}
        if torch_dtype is not torch.float32:
            kwargs["torch_dtype"] = torch_dtype

        self._processor = AutoProcessor.from_pretrained(self.entry.hf_repo_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.entry.hf_repo_id, **kwargs
        )
        self._model.to(device)
        self._model.eval()
        self._torch = torch

    def warmup(self) -> None:
        if not self._real_ready:
            return
        try:
            dummy = Image.new("RGB", (64, 64), color="black")
            self.predict(dummy, prompts=["object"])
        except Exception:
            pass

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
        threshold: float = 0.3,
        text_threshold: float = 0.25,
        **kwargs: Any,
    ) -> BaseResult:
        if not self._real_ready:
            return super().predict(image, prompts=prompts, **kwargs)

        if not prompts:
            prompts = ["object"]
        prompt_text = ". ".join(p.strip().rstrip(".") for p in prompts) + "."

        inputs = self._processor(
            images=image, text=prompt_text, return_tensors="pt"
        ).to(self._model.device)

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        # Recent versions of transformers renamed ``box_threshold`` → ``threshold``.
        post = self._processor.post_process_grounded_object_detection
        try:
            results = post(
                outputs,
                inputs.input_ids,
                threshold=threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )
        except TypeError:
            results = post(
                outputs,
                inputs.input_ids,
                box_threshold=threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )
        first = results[0] if results else {"boxes": [], "scores": [], "labels": []}

        # Newer transformers return the predicted-class strings under
        # ``labels`` or ``text_labels`` depending on version.
        labels_iter = first.get("labels", None)
        if labels_iter is None:
            labels_iter = first.get("text_labels", [])

        detections: list[Detection] = []
        for box, score, label in zip(first["boxes"], first["scores"], labels_iter):
            xyxy = [float(v) for v in (box.tolist() if hasattr(box, "tolist") else box)]
            detections.append(
                Detection(
                    box=Box(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]),
                    score=float(score),
                    label=str(label),
                    class_id=None,
                )
            )

        return OpenVocabularyResult(
            kind="open_vocab",
            model_id=self.entry.id,
            task=self.entry.task,
            image_size=image.size,
            device=self.device,
            prompts=list(prompts),
            detections=detections,
            metadata={"backend": self.backend_label, "precision": self.precision},
        )


def _factory(entry: ModelEntry) -> GroundingDINOEngine:
    return GroundingDINOEngine(entry)


register_engine("grounding_dino", _factory)

__all__ = ["GroundingDINOEngine"]
