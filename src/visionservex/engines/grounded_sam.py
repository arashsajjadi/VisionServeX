# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Grounded SAM composed pipeline engine.

Combines Grounding DINO (text-prompted object detection) with SAM v1
(box-prompted mask generation) into a single grounded segmentation pipeline.

Grounding DINO locates objects from text prompts → produces bounding boxes.
SAM v1 segments each box → produces per-object masks.

Install:
    pip install 'visionservex[grounding]'   # also includes hf via dependencies

Model ID:
    grounded-sam
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from PIL import Image

from visionservex.core.results import (
    BaseResult,
    Box,
    Detection,
    Segment,
    SegmentationResult,
)
from visionservex.engines._stub import StubEngine
from visionservex.engines.base import BaseEngine, MissingDependencyError
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry, default_registry
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)

# Default sub-model IDs used by the composed pipeline
_GD_MODEL_ID = "grounding-dino-tiny"
_SAM_MODEL_ID = "sam-vit-base"


class GroundedSAMEngine(StubEngine):
    """Composes Grounding DINO + SAM v1 for text-prompted segmentation.

    The engine loads both sub-models lazily and reuses them across requests
    within the same process. Both models must be downloadable via HF.
    """

    real_install_extra = "grounding"
    real_modules = ("transformers", "torch")
    backend_label = "composed_gd_sam"

    def __init__(self, entry: ModelEntry) -> None:
        super().__init__(entry)
        self._gd_engine: Any = None
        self._sam_engine: Any = None

    def _real_load(self, *, device: str, precision: str) -> None:
        from visionservex.engines.grounding_dino import GroundingDINOEngine
        from visionservex.engines.sam_hf import SAMHFEngine

        try:
            gd_entry = default_registry().get(_GD_MODEL_ID)
        except Exception as exc:
            raise MissingDependencyError(
                f"Grounding DINO sub-model {_GD_MODEL_ID!r} not in registry",
                install_hint="run `visionservex list-models`",
            ) from exc
        try:
            sam_entry = default_registry().get(_SAM_MODEL_ID)
        except Exception as exc:
            raise MissingDependencyError(
                f"SAM sub-model {_SAM_MODEL_ID!r} not in registry",
                install_hint="run `visionservex list-models`",
            ) from exc

        _log.info("loading GroundedSAM sub-models on device=%s", device)
        self._gd_engine = GroundingDINOEngine(gd_entry)
        self._gd_engine.load(device=device, precision=precision)

        self._sam_engine = SAMHFEngine(sam_entry)
        self._sam_engine.load(device=device, precision=precision)
        _log.info("GroundedSAM ready")

    def unload(self) -> None:
        if self._gd_engine is not None:
            self._gd_engine.unload()
        if self._sam_engine is not None:
            self._sam_engine.unload()
        self._gd_engine = None
        self._sam_engine = None
        super().unload()

    def warmup(self) -> None:
        if not self._real_ready:
            return
        try:
            dummy = Image.new("RGB", (128, 128), "gray")
            self.predict(dummy, prompts=["object"])
        except Exception:
            pass

    def predict(
        self,
        image: Image.Image,
        *,
        prompts: Sequence[str] | None = None,
        detection_threshold: float = 0.3,
        **kwargs: Any,
    ) -> BaseResult:
        if not self._real_ready:
            return super().predict(image, prompts=prompts, **kwargs)

        if not prompts:
            prompts = ["object"]

        w, h = image.size

        # Step 1: Grounding DINO → bounding boxes
        gd_result = self._gd_engine.predict(
            image,
            prompts=list(prompts),
            threshold=detection_threshold,
        )
        detections: list[Detection] = getattr(gd_result, "detections", [])

        # Step 2: SAM → mask per box
        segments: list[Segment] = []
        for det in detections:
            box = det.box
            try:
                sam_result = self._sam_engine.predict(
                    image,
                    boxes=[[box.x1, box.y1, box.x2, box.y2]],
                )
                for seg in sam_result.segments:
                    segments.append(Segment(
                        box=det.box,
                        score=det.score,
                        label=det.label,
                        mask=seg.mask,
                        class_id=det.class_id,
                    ))
            except Exception as exc:
                _log.warning("SAM failed for box %s: %s", det.box, exc)
                segments.append(Segment(
                    box=det.box,
                    score=det.score,
                    label=det.label,
                    mask=np.zeros((h, w), dtype=np.uint8),
                    class_id=det.class_id,
                ))

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


def _factory(entry: ModelEntry) -> GroundedSAMEngine:
    return GroundedSAMEngine(entry)


register_engine("grounded_sam", _factory)

__all__ = ["GroundedSAMEngine"]
