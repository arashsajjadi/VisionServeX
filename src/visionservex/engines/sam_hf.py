# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""SAM v1 (Segment Anything) engine via HF Transformers.

Uses ``SamModel`` and ``SamProcessor`` from HuggingFace transformers.
Supports point prompts and box prompts.

Install:
    pip install 'visionservex[hf]'

Supported model IDs:
    sam-vit-base, sam-vit-large, sam-vit-huge
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from visionservex.core.results import (
    BaseResult,
    Box,
    Segment,
    SegmentationResult,
)
from visionservex.engines._stub import StubEngine
from visionservex.engines.base import MissingDependencyError
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)


class SAMHFEngine(StubEngine):
    """HF Transformers-backed SAM v1 engine.

    Prompt interface:
    - ``points`` kwarg: list of [x, y] pairs
    - ``point_labels`` kwarg: list of 0/1 (background/foreground)
    - ``boxes`` kwarg: list of [x1, y1, x2, y2] tuples

    If no prompts are given, SAM generates masks from a uniform grid of points
    across the image (automatic mask generation mode, limited to center point).
    """

    real_install_extra = "hf"
    real_modules = ("transformers", "torch")
    backend_label = "huggingface_sam"

    def __init__(self, entry: ModelEntry) -> None:
        super().__init__(entry)
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

    def _real_load(self, *, device: str, precision: str) -> None:
        if not self.entry.hf_repo_id:
            raise MissingDependencyError(
                f"model {self.entry.id!r} has no hf_repo_id",
                install_hint=self._install_hint(),
            )
        from visionservex.runtime.downloads import download

        download(self.entry)

        import torch  # type: ignore
        from transformers import SamModel, SamProcessor  # type: ignore

        torch_dtype = torch.float32
        if precision in ("fp16", "bf16") and device != "cpu":
            torch_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

        kwargs: dict[str, Any] = {}
        if torch_dtype is not torch.float32:
            kwargs["torch_dtype"] = torch_dtype

        _log.info("loading %s from %s", self.entry.id, self.entry.hf_repo_id)
        self._processor = SamProcessor.from_pretrained(self.entry.hf_repo_id)
        self._model = SamModel.from_pretrained(self.entry.hf_repo_id, **kwargs)
        self._model.to(device)
        self._model.eval()
        self._torch = torch
        _log.info("%s loaded", self.entry.id)

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
            dummy = Image.new("RGB", (128, 128), "gray")
            self.predict(dummy, points=[[64, 64]], point_labels=[1])
        except Exception:
            pass

    def predict(
        self,
        image: Image.Image,
        *,
        prompts: Sequence[str] | None = None,
        points: Sequence[Sequence[float]] | None = None,
        point_labels: Sequence[int] | None = None,
        boxes: Sequence[Sequence[float]] | None = None,
        multimask_output: bool = True,
        **kwargs: Any,
    ) -> BaseResult:
        if not self._real_ready:
            return super().predict(image, prompts=prompts, **kwargs)

        w, h = image.size

        # Default prompt: single center point if nothing provided
        if not points and not boxes:
            points = [[w // 2, h // 2]]
            point_labels = [1]

        # Build processor kwargs
        proc_kwargs: dict[str, Any] = {"images": image, "return_tensors": "pt"}
        if points is not None:
            proc_kwargs["input_points"] = [[[float(p[0]), float(p[1])] for p in points]]
        if point_labels is not None:
            proc_kwargs["input_labels"] = [list(point_labels)]
        if boxes is not None:
            proc_kwargs["input_boxes"] = [
                [[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in boxes]
            ]

        inputs = self._processor(**proc_kwargs)

        model_device = next(self._model.parameters()).device
        model_dtype = next(self._model.parameters()).dtype
        inputs_dev: dict[str, Any] = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                v = v.to(device=model_device)
                if hasattr(v, "is_floating_point") and v.is_floating_point():
                    v = v.to(dtype=model_dtype)
            inputs_dev[k] = v

        with self._torch.no_grad():
            outputs = self._model(**inputs_dev)

        # Post-process masks
        masks_raw = self._processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        # masks_raw[0] shape: (num_prompts, num_masks, H, W) bool tensor. There is
        # one prompt per box (input_boxes=(1, N, 4)); points are combined into a
        # single prompt, so points-only stays one prompt.
        masks_tensor = masks_raw[0]
        if masks_tensor.ndim == 3:  # (num_masks, H, W) -> single prompt
            masks_tensor = masks_tensor.unsqueeze(0)
        iou_scores = outputs.iou_scores.cpu()  # (1, num_prompts, num_masks)

        # v3.23 fix: emit ONE segment per prompt (box), not just the first. The
        # previous code took masks_tensor[0] and returned a single mask even when
        # multiple boxes were supplied, silently dropping all but the first box.
        num_prompts = masks_tensor.shape[0]
        segments: list[Segment] = []
        for p in range(num_prompts):
            prompt_masks = masks_tensor[p]  # (num_masks, H, W)
            prompt_iou = iou_scores[0, p]  # (num_masks,)
            best_idx = int(prompt_iou.argmax())
            mask_uint8 = prompt_masks[best_idx].numpy().astype(np.uint8)
            # Tight bounding box from the predicted mask; fall back to the input
            # box (when given) or the full frame for an empty mask.
            ys, xs = np.where(mask_uint8)
            if len(xs) > 0:
                box = Box(
                    x1=float(xs.min()),
                    y1=float(ys.min()),
                    x2=float(xs.max()),
                    y2=float(ys.max()),
                )
                score = float(prompt_iou[best_idx])
            elif boxes is not None and p < len(boxes):
                b = boxes[p]
                box = Box(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]))
                score = 0.0
            else:
                box = Box(0, 0, float(w), float(h))
                score = 0.0
            segments.append(Segment(box=box, score=score, label="segment", mask=mask_uint8))

        return SegmentationResult(
            kind="segmentation",
            model_id=self.entry.id,
            task=self.entry.task,
            image_size=(w, h),
            device=self.device,
            precision=self.precision,
            backend=self.backend_label,
            segments=segments,
        )

    def infer(self, preprocessed: Any, **kwargs: Any) -> Any:
        return preprocessed

    def postprocess(self, raw: Any, *, image: Any, **kwargs: Any) -> BaseResult:
        return self._mock.postprocess(raw, image=image, **kwargs)


def _factory(entry: ModelEntry) -> SAMHFEngine:
    return SAMHFEngine(entry)


register_engine("sam_hf", _factory)

__all__ = ["SAMHFEngine"]
