# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Florence-2 — multi-task vision-language model engine.

Real implementation via Hugging Face Transformers with ``trust_remote_code=True``:
``AutoProcessor`` + ``AutoModelForCausalLM``.

Reference:
- https://github.com/microsoft/Florence-2
- https://huggingface.co/microsoft/Florence-2-base
- https://huggingface.co/microsoft/Florence-2-large

Supported tasks (via the ``task`` kwarg or auto-mapping from prompt tokens):
- caption                       → "<CAPTION>"
- detailed_caption              → "<DETAILED_CAPTION>"
- more_detailed_caption         → "<MORE_DETAILED_CAPTION>"
- object_detection              → "<OD>"
- dense_region_caption          → "<DENSE_REGION_CAPTION>"
- phrase_grounding              → "<CAPTION_TO_PHRASE_GROUNDING>"
- ocr                           → "<OCR>"
- region_ocr / ocr_with_region  → "<OCR_WITH_REGION>"

Output: a single ``OpenVocabularyResult`` whose ``detections`` list is populated
for box-producing tasks; text-only tasks (caption, ocr) put the generated text
in ``metadata["text"]`` and leave ``detections`` empty.
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

# Task → prompt token table.
_TASK_TOKEN: dict[str, str] = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    "object_detection": "<OD>",
    "dense_caption": "<DENSE_REGION_CAPTION>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
    "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "ocr": "<OCR>",
    "region_ocr": "<OCR_WITH_REGION>",
    "ocr_with_region": "<OCR_WITH_REGION>",
}


def parse_florence2_generation(
    parsed: Any, *, task_token: str, image_size: tuple[int, int]
) -> tuple[list[Detection], dict[str, Any]]:
    """Convert Florence-2's parsed generation output into (detections, extra_metadata).

    The processor's ``post_process_generation`` returns a dict keyed by task token.
    For box-producing tasks the value is a dict with ``bboxes`` and ``labels``.
    For text-only tasks the value is a string.
    Phrase grounding and dense region caption produce both boxes and per-box labels.
    """
    detections: list[Detection] = []
    extra: dict[str, Any] = {"task_token": task_token}

    if not isinstance(parsed, dict):
        extra["raw"] = str(parsed)[:500]
        return detections, extra

    payload = parsed.get(task_token, parsed)

    # Text-only tasks
    if isinstance(payload, str):
        extra["text"] = payload
        return detections, extra

    if not isinstance(payload, dict):
        extra["raw"] = str(payload)[:500]
        return detections, extra

    # Box-producing tasks: keys we recognise across upstream versions
    bboxes = payload.get("bboxes") or payload.get("boxes") or payload.get("quad_boxes") or []
    labels = payload.get("labels") or payload.get("bboxes_labels") or []

    for idx, box in enumerate(bboxes):
        # Polygons (quad_boxes) → convert to axis-aligned bbox
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            if len(box) == 4:
                x1, y1, x2, y2 = (float(v) for v in box)
            else:
                xs = [float(box[i]) for i in range(0, len(box), 2)]
                ys = [float(box[i]) for i in range(1, len(box), 2)]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            continue

        label = str(labels[idx]) if idx < len(labels) else ""
        detections.append(
            Detection(
                box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                score=1.0,  # Florence-2 generative output has no per-box score
                label=label,
                class_id=None,
            )
        )

    if not detections and not labels:
        # Some upstream versions return only flat strings
        extra["raw"] = str(payload)[:500]
    return detections, extra


class Florence2Engine(StubEngine):
    """Florence-2 multi-task VLM engine."""

    real_install_extra = "hf"
    real_modules = ("transformers", "torch")
    backend_label = "huggingface_florence2"

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
        from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore

        torch_dtype = torch.float32
        if precision == "fp16" and device != "cpu":
            torch_dtype = torch.float16
        elif precision == "bf16" and device != "cpu":
            torch_dtype = torch.bfloat16

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if torch_dtype is not torch.float32:
            kwargs["torch_dtype"] = torch_dtype

        self._processor = AutoProcessor.from_pretrained(
            self.entry.hf_repo_id, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(self.entry.hf_repo_id, **kwargs)
        self._model.to(device)
        self._model.eval()
        self._torch = torch

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
        task: str | None = None,
        max_new_tokens: int = 1024,
        **kwargs: Any,
    ) -> BaseResult:
        if not self._real_ready:
            return super().predict(image, prompts=prompts, prompt=prompt, task=task, **kwargs)

        # Resolve task → token.
        task_key = (task or "caption").lower().strip()
        task_token = _TASK_TOKEN.get(task_key)
        if task_token is None:
            raise ValueError(
                f"Florence-2 task {task!r} unsupported. Choose from: {sorted(_TASK_TOKEN.keys())}"
            )

        # For phrase grounding, the user's prompt is appended after the token.
        user_text = ""
        if prompt:
            user_text = str(prompt).strip()
        elif prompts:
            user_text = ", ".join(str(p).strip() for p in prompts).strip()

        prompt_text = task_token
        if task_token == "<CAPTION_TO_PHRASE_GROUNDING>" and user_text:
            prompt_text = f"{task_token} {user_text}"

        inputs = self._processor(text=prompt_text, images=image, return_tensors="pt")
        model_device = next(self._model.parameters()).device
        model_dtype = next(self._model.parameters()).dtype
        inputs_on_device = {}
        for k, v in inputs.items():
            v = v.to(device=model_device)
            if v.is_floating_point():
                v = v.to(dtype=model_dtype)
            inputs_on_device[k] = v

        with self._torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs_on_device.get("input_ids"),
                pixel_values=inputs_on_device.get("pixel_values"),
                max_new_tokens=max_new_tokens,
                num_beams=3,
                do_sample=False,
            )

        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        try:
            parsed = self._processor.post_process_generation(
                generated_text,
                task=task_token,
                image_size=image.size,
            )
        except Exception as exc:
            _log.warning("Florence-2 post_process_generation failed: %s", exc)
            parsed = {"raw": generated_text}

        detections, extra = parse_florence2_generation(
            parsed, task_token=task_token, image_size=image.size
        )
        prompts_list = [user_text] if user_text else []

        metadata: dict[str, Any] = {
            "backend": self.backend_label,
            "precision": self.precision,
            "task": task_key,
            "task_token": task_token,
        }
        metadata.update(extra)

        return OpenVocabularyResult(
            kind="open_vocab",
            model_id=self.entry.id,
            task=task_key,
            image_size=image.size,
            device=self.device,
            prompts=prompts_list,
            detections=detections,
            metadata=metadata,
        )


def _factory(entry: ModelEntry) -> Florence2Engine:
    return Florence2Engine(entry)


register_engine("florence2", _factory)

__all__ = ["Florence2Engine", "parse_florence2_generation"]
