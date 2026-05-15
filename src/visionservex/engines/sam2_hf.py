# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""SAM 2 / SAM 2.1 engine via Hugging Face Transformers.

Uses ``Sam2Model`` + ``Sam2Processor`` from the official Facebook SAM2
checkpoints on HuggingFace. This is NOT the same as the upstream
``sam2`` pip package — it uses the transformers implementation which
does not require building CUDA extensions.

Supported model IDs:
    sam2-hiera-tiny       → facebook/sam2-hiera-tiny
    sam2-hiera-small      → facebook/sam2-hiera-small
    sam2-hiera-base-plus  → facebook/sam2-hiera-base-plus
    sam2-hiera-large      → facebook/sam2-hiera-large

Input prompt format:
    points: list of [x, y] pairs (image pixel coordinates)
    point_labels: list of int (1=foreground, 0=background)
    boxes:  list of [x1, y1, x2, y2] tuples

Install:
    pip install 'visionservex[sam2]'   (or [hf])
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from visionservex.core.results import BaseResult, Box, Segment, SegmentationResult
from visionservex.engines._stub import StubEngine
from visionservex.engines.base import MissingDependencyError
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)

_HF_REPOS: dict[str, str] = {
    "sam2-hiera-tiny": "facebook/sam2-hiera-tiny",
    "sam2-hiera-small": "facebook/sam2-hiera-small",
    "sam2-hiera-base-plus": "facebook/sam2-hiera-base-plus",
    "sam2-hiera-large": "facebook/sam2-hiera-large",
}


class SAM2HFEngine(StubEngine):
    """HF Transformers SAM2 engine. Works without CUDA extensions."""

    real_install_extra = "sam2"
    real_modules = ("transformers", "torch")
    backend_label = "huggingface_sam2"

    def __init__(self, entry: ModelEntry) -> None:
        super().__init__(entry)
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

    def _real_load(self, *, device: str, precision: str) -> None:
        repo = self.entry.hf_repo_id or _HF_REPOS.get(self.entry.id)
        if not repo:
            raise MissingDependencyError(
                f"no HF repo for SAM2 model {self.entry.id!r}",
                install_hint="check `visionservex info sam2-hiera-tiny`",
            )
        if not self.entry.hf_repo_id:
            self.entry.hf_repo_id = repo  # type: ignore[misc]

        from visionservex.runtime.downloads import download

        download(self.entry)

        import torch  # type: ignore
        from transformers import Sam2Model, Sam2Processor  # type: ignore

        torch_dtype = torch.float32
        if precision in ("fp16", "bf16") and device != "cpu":
            torch_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

        kwargs: dict[str, Any] = {}
        if torch_dtype is not torch.float32:
            kwargs["torch_dtype"] = torch_dtype

        _log.info("loading SAM2 %s from %s on %s", self.entry.id, repo, device)
        self._processor = Sam2Processor.from_pretrained(repo)
        self._model = Sam2Model.from_pretrained(repo, **kwargs)
        self._model.to(device)
        self._model.eval()
        self._torch = torch
        _log.info("SAM2 %s ready", self.entry.id)

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
        **kwargs: Any,
    ) -> BaseResult:
        if not self._real_ready:
            return super().predict(image, prompts=prompts, **kwargs)

        w, h = image.size

        # Default to center point if no prompt given
        if not points and not boxes:
            points = [[w // 2, h // 2]]
            point_labels = [1]

        # SAM2 processor expects 4-level nesting for points:
        # [image_level, object_level, point_level, coords]
        proc_kwargs: dict[str, Any] = {"images": image, "return_tensors": "pt"}
        if points is not None:
            proc_kwargs["input_points"] = [[[[float(p[0]), float(p[1])] for p in points]]]
        if point_labels is not None:
            proc_kwargs["input_labels"] = [[[int(lbl) for lbl in point_labels]]]
        if boxes is not None:
            # SAM2 box format: [image_level, box_level, coords] = 3 levels
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
            out = self._model(**inputs_dev)

        masks_out = self._processor.post_process_masks(
            out.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
        )
        # masks_out[0]: (1, num_masks, H, W) bool tensor
        iou_scores = out.iou_scores.cpu()  # (1, 1, num_masks)

        segments: list[Segment] = []
        if masks_out and masks_out[0].shape[0] > 0:
            batch_masks = masks_out[0][0]  # (num_masks, H, W)
            batch_iou = iou_scores[0, 0]  # (num_masks,)
            best_idx = int(batch_iou.argmax())
            mask_bool = batch_masks[best_idx].numpy()
            mask_uint8 = mask_bool.astype(np.uint8)
            ys, xs = np.where(mask_uint8)
            if len(xs) > 0:
                box = Box(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
                score = float(batch_iou[best_idx])
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


def _factory(entry: ModelEntry) -> SAM2HFEngine:
    return SAM2HFEngine(entry)


register_engine("sam2_hf", _factory)

__all__ = ["SAM2HFEngine"]
