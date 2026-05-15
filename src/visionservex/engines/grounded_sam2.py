# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Grounded-SAM2 composed pipeline: Grounding DINO + SAM 2 (HF).

Grounding DINO (HF Transformers) detects objects from text prompts and
returns bounding boxes. SAM 2 (HF) then segments each box to produce
per-object masks.

This is the SAM2 version of the ``grounded-sam`` pipeline. Both sub-models
are loaded lazily and reused across requests via the shared engine cache.

Model ID:
    grounded-sam2

Install:
    pip install 'visionservex[hf]'
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from visionservex.core.results import (
    BaseResult,
    Segment,
    SegmentationResult,
)
from visionservex.engines._stub import StubEngine
from visionservex.engines.base import MissingDependencyError
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry, default_registry
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)

_GD_MODEL_ID = "grounding-dino-tiny"
_SAM2_MODEL_ID = "sam2-hiera-tiny"


class GroundedSAM2Engine(StubEngine):
    """Composes Grounding DINO + SAM2 (HF) for text-prompted instance masks.

    Output metadata includes:
        detector_model_id, segmenter_model_id, detector_device, segmenter_device
    """

    real_install_extra = "hf"
    real_modules = ("transformers", "torch")
    backend_label = "composed_gd_sam2"

    def __init__(self, entry: ModelEntry) -> None:
        super().__init__(entry)
        self._gd_engine: Any = None
        self._sam2_engine: Any = None

    def _real_load(self, *, device: str, precision: str) -> None:
        from visionservex.engines.grounding_dino import GroundingDINOEngine
        from visionservex.engines.sam2_hf import SAM2HFEngine

        reg = default_registry()
        try:
            gd_entry = reg.get(_GD_MODEL_ID)
        except Exception as exc:
            raise MissingDependencyError(
                f"Grounding DINO sub-model {_GD_MODEL_ID!r} not found",
                install_hint="run `visionservex list-models`",
            ) from exc
        try:
            sam2_entry = reg.get(_SAM2_MODEL_ID)
        except Exception as exc:
            raise MissingDependencyError(
                f"SAM2 sub-model {_SAM2_MODEL_ID!r} not found",
                install_hint="run `visionservex list-models`",
            ) from exc

        _log.info("loading GroundedSAM2 sub-models on device=%s", device)
        self._gd_engine = GroundingDINOEngine(gd_entry)
        self._gd_engine.load(device=device, precision=precision)

        self._sam2_engine = SAM2HFEngine(sam2_entry)
        self._sam2_engine.load(device=device, precision=precision)
        _log.info("GroundedSAM2 ready (GD+SAM2)")

    def unload(self) -> None:
        if self._gd_engine is not None:
            self._gd_engine.unload()
        if self._sam2_engine is not None:
            self._sam2_engine.unload()
        self._gd_engine = None
        self._sam2_engine = None
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

        # Step 1: Grounding DINO → boxes
        gd_result = self._gd_engine.predict(
            image,
            prompts=list(prompts),
            threshold=detection_threshold,
        )
        detections = getattr(gd_result, "detections", [])

        # Step 2: SAM2 → mask per box
        segments: list[Segment] = []
        for det in detections:
            box = det.box
            try:
                sam_result = self._sam2_engine.predict(
                    image,
                    boxes=[[box.x1, box.y1, box.x2, box.y2]],
                )
                for seg in sam_result.segments:
                    segments.append(
                        Segment(
                            box=det.box,
                            score=det.score,
                            label=det.label,
                            mask=seg.mask,
                            class_id=det.class_id,
                        )
                    )
            except Exception as exc:
                _log.warning("SAM2 failed for box %s: %s", det.box, exc)
                segments.append(
                    Segment(
                        box=det.box,
                        score=det.score,
                        label=det.label,
                        mask=np.zeros((h, w), dtype=np.uint8),
                        class_id=det.class_id,
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
            metadata={
                "detector_model_id": _GD_MODEL_ID,
                "segmenter_model_id": _SAM2_MODEL_ID,
                "detector_device": self.device,
                "segmenter_device": self.device,
                "prompts": list(prompts),
            },
        )

    def infer(self, preprocessed: Any, **kwargs: Any) -> Any:
        return preprocessed

    def postprocess(self, raw: Any, *, image: Any, **kwargs: Any) -> BaseResult:
        return self._mock.postprocess(raw, image=image, **kwargs)


def _factory(entry: ModelEntry) -> GroundedSAM2Engine:
    return GroundedSAM2Engine(entry)


register_engine("grounded_sam2", _factory)

__all__ = ["GroundedSAM2Engine"]
