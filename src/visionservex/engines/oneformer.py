# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""OneFormer universal segmentation engine via HF Transformers.

Supports semantic, instance, and panoptic segmentation through a single model.
Uses ``OneFormerProcessor`` + ``OneFormerForUniversalSegmentation``.

Supported model IDs:
    oneformer-swin-large       → shi-labs/oneformer_coco_swin_large
    oneformer-dinat-large      → shi-labs/oneformer_coco_dinat_large
    oneformer-convnext-large   → shi-labs/oneformer_ade20k_convnext_large

Install:
    pip install 'visionservex[hf]'

Default segmentation task is semantic. Pass ``task`` kwarg to predict()
to override: ``task='instance'`` or ``task='panoptic'``.
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
    "oneformer-swin-large": "shi-labs/oneformer_coco_swin_large",
    "oneformer-dinat-large": "shi-labs/oneformer_coco_dinat_large",
    "oneformer-convnext-large": "shi-labs/oneformer_ade20k_convnext_large",
}


class OneFormerEngine(StubEngine):
    """Real OneFormer engine backed by HF Transformers."""

    real_install_extra = "hf"
    real_modules = ("transformers", "torch")
    backend_label = "huggingface_oneformer"

    def __init__(self, entry: ModelEntry) -> None:
        super().__init__(entry)
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._id2label: dict[int, str] = {}

    def _real_load(self, *, device: str, precision: str) -> None:
        repo = self.entry.hf_repo_id or _HF_REPOS.get(self.entry.id)
        if not repo:
            raise MissingDependencyError(
                f"no HF repo for OneFormer model {self.entry.id!r}",
                install_hint="check `visionservex info oneformer-swin-large`",
            )
        if not self.entry.hf_repo_id:
            self.entry.hf_repo_id = repo  # type: ignore[misc]

        from visionservex.runtime.downloads import download

        download(self.entry)

        import torch  # type: ignore
        from transformers import (  # type: ignore
            OneFormerForUniversalSegmentation,
            OneFormerProcessor,
        )

        torch_dtype = torch.float32
        if precision in ("fp16", "bf16") and device != "cpu":
            torch_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

        kwargs: dict[str, Any] = {}
        if torch_dtype is not torch.float32:
            kwargs["torch_dtype"] = torch_dtype

        _log.info("loading OneFormer %s from %s on %s", self.entry.id, repo, device)
        self._processor = OneFormerProcessor.from_pretrained(repo)
        self._model = OneFormerForUniversalSegmentation.from_pretrained(repo, **kwargs)
        self._model.to(device)
        self._model.eval()
        self._torch = torch

        cfg = self._model.config
        self._id2label = {int(k): v for k, v in (getattr(cfg, "id2label", {}) or {}).items()}
        _log.info("OneFormer %s ready (%d classes)", self.entry.id, len(self._id2label))

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
            self.predict(dummy, task="semantic")
        except Exception:
            pass

    def predict(
        self,
        image: Image.Image,
        *,
        prompts: Sequence[str] | None = None,
        task: str = "semantic",
        **kwargs: Any,
    ) -> BaseResult:
        if not self._real_ready:
            return super().predict(image, prompts=prompts, **kwargs)

        valid_tasks = {"semantic", "instance", "panoptic"}
        if task not in valid_tasks:
            _log.warning("OneFormer task %r not recognised; using 'semantic'", task)
            task = "semantic"

        w, h = image.size
        model_device = next(self._model.parameters()).device
        model_dtype = next(self._model.parameters()).dtype

        inputs = self._processor(images=image, task_inputs=[task], return_tensors="pt")
        inputs_dev: dict[str, Any] = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                v = v.to(device=model_device)
                if hasattr(v, "is_floating_point") and v.is_floating_point():
                    v = v.to(dtype=model_dtype)
            inputs_dev[k] = v

        with self._torch.no_grad():
            out = self._model(**inputs_dev)

        segments: list[Segment] = []

        if task == "semantic":
            result = self._processor.post_process_semantic_segmentation(out, target_sizes=[(h, w)])
            seg_map = result[0].cpu().numpy().astype(np.int32)  # (H, W)
            unique_ids = np.unique(seg_map)
            for uid in unique_ids:
                label = self._id2label.get(int(uid), f"class_{uid}")
                mask = (seg_map == uid).astype(np.uint8)
                ys, xs = np.where(mask)
                box = Box(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
                area_frac = float(mask.sum()) / (h * w)
                segments.append(
                    Segment(
                        box=box,
                        score=area_frac,
                        label=label,
                        mask=mask,
                        class_id=int(uid),
                    )
                )

        elif task == "instance":
            result = self._processor.post_process_instance_segmentation(out, target_sizes=[(h, w)])
            info = result[0]
            seg_map = info.get("segmentation")
            if seg_map is not None:
                seg_map = seg_map.cpu().numpy().astype(np.int32)
                for seg_info in info.get("segments_info", []):
                    sid = seg_info.get("id", 0)
                    label_id = seg_info.get("label_id", 0)
                    label = self._id2label.get(int(label_id), f"class_{label_id}")
                    score = seg_info.get("score", 1.0) or 1.0
                    mask = (seg_map == sid).astype(np.uint8)
                    ys, xs = np.where(mask)
                    if len(xs) == 0:
                        continue
                    box = Box(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
                    segments.append(
                        Segment(
                            box=box,
                            score=float(score),
                            label=label,
                            mask=mask,
                            class_id=int(label_id),
                        )
                    )

        elif task == "panoptic":
            result = self._processor.post_process_panoptic_segmentation(out, target_sizes=[(h, w)])
            info = result[0]
            seg_map = info.get("segmentation")
            if seg_map is not None:
                seg_map = seg_map.cpu().numpy().astype(np.int32)
                for seg_info in info.get("segments_info", []):
                    sid = seg_info.get("id", 0)
                    label_id = seg_info.get("label_id", 0)
                    label = self._id2label.get(int(label_id), f"class_{label_id}")
                    score = seg_info.get("score", 1.0) or 1.0
                    mask = (seg_map == sid).astype(np.uint8)
                    ys, xs = np.where(mask)
                    if len(xs) == 0:
                        continue
                    box = Box(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
                    segments.append(
                        Segment(
                            box=box,
                            score=float(score),
                            label=label,
                            mask=mask,
                            class_id=int(label_id),
                        )
                    )

        return SegmentationResult(
            kind="segmentation",
            model_id=self.entry.id,
            task=self.entry.task,
            image_size=(w, h),
            device=self.device,
            precision=self.precision,
            backend=self.backend_label,
            segments=segments,
            metadata={"oneformer_task": task},
        )

    def infer(self, preprocessed: Any, **kwargs: Any) -> Any:
        return preprocessed

    def postprocess(self, raw: Any, *, image: Any, **kwargs: Any) -> BaseResult:
        return self._mock.postprocess(raw, image=image, **kwargs)


def _factory(entry: ModelEntry) -> OneFormerEngine:
    return OneFormerEngine(entry)


register_engine("oneformer", _factory)

__all__ = ["OneFormerEngine"]
