"""Deterministic fake engine.

Used by tests and as a friendly default for users who run VisionServeX
without any heavy backend installed. The output is shaped like real engine
output so the rest of the system can be exercised end-to-end.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from visionservex.core.results import (
    BaseResult,
    Box,
    ClassificationResult,
    Detection,
    DetectionResult,
    Keypoint,
    OpenVocabularyResult,
    OrientedBox,
    OrientedDetection,
    OrientedDetectionResult,
    PoseInstance,
    PoseResult,
    Segment,
    SegmentationResult,
)
from visionservex.engines.base import BaseEngine
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry


class MockEngine(BaseEngine):
    """Deterministic dummy engine. Produces stable, image-derived outputs."""

    def load(self, *, device: str, precision: str) -> None:
        self.device = device
        self.precision = precision
        self._loaded = True

    def warmup(self) -> None:
        return None

    def infer(self, preprocessed: Any, **kwargs: Any) -> Any:
        return preprocessed  # the PIL image is the "raw output"

    def postprocess(
        self,
        raw: Any,
        *,
        image: Image.Image,
        prompts: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> BaseResult:
        task = self.entry.task
        seed = _seed_from_image(image)
        rng = np.random.default_rng(seed)
        w, h = image.size

        if task == "detect":
            return DetectionResult(
                kind="detection",
                model_id=self.entry.id,
                task=task,
                image_size=(w, h),
                detections=_fake_detections(rng, w, h),
            )
        if task == "segment":
            return SegmentationResult(
                kind="segmentation",
                model_id=self.entry.id,
                task=task,
                image_size=(w, h),
                segments=_fake_segments(rng, w, h),
            )
        if task == "pose":
            return PoseResult(
                kind="pose",
                model_id=self.entry.id,
                task=task,
                image_size=(w, h),
                persons=_fake_pose(rng, w, h),
            )
        if task == "classify":
            return ClassificationResult(
                kind="classification",
                model_id=self.entry.id,
                task=task,
                image_size=(w, h),
                top_k=[("cat", 0.62), ("dog", 0.21), ("bird", 0.05)],
            )
        if task == "obb":
            return OrientedDetectionResult(
                kind="obb",
                model_id=self.entry.id,
                task=task,
                image_size=(w, h),
                detections=_fake_obb(rng, w, h),
            )
        if task == "open_vocab_detect":
            labels = list(prompts) if prompts else ["object"]
            dets = _fake_detections(rng, w, h, labels=labels)
            return OpenVocabularyResult(
                kind="open_vocab",
                model_id=self.entry.id,
                task=task,
                image_size=(w, h),
                prompts=list(prompts) if prompts else [],
                detections=dets,
            )
        if task in {"grounded_segment", "foundation_segment"}:
            labels = list(prompts) if prompts else ["object"]
            return SegmentationResult(
                kind="segmentation",
                model_id=self.entry.id,
                task=task,
                image_size=(w, h),
                segments=_fake_segments(rng, w, h, labels=labels),
            )

        raise NotImplementedError(f"mock engine does not know task {task!r}")


def _seed_from_image(image: Image.Image) -> int:
    h = hashlib.sha1(image.tobytes()[:1024]).digest()
    return int.from_bytes(h[:4], "little")


def _fake_detections(
    rng: np.random.Generator,
    w: int,
    h: int,
    *,
    labels: list[str] | None = None,
) -> list[Detection]:
    labels = labels or ["person", "car", "dog"]
    n = int(rng.integers(1, 4))
    dets: list[Detection] = []
    for i in range(n):
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        bw, bh = rng.uniform(0.1, 0.3) * w, rng.uniform(0.1, 0.3) * h
        box = Box(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        dets.append(
            Detection(
                box=box,
                score=float(rng.uniform(0.5, 0.95)),
                label=labels[i % len(labels)],
                class_id=i % len(labels),
            )
        )
    return dets


def _fake_segments(
    rng: np.random.Generator,
    w: int,
    h: int,
    *,
    labels: list[str] | None = None,
) -> list[Segment]:
    labels = labels or ["object"]
    n = 1
    segs: list[Segment] = []
    for i in range(n):
        cx, cy = w // 2, h // 2
        rad = max(8, min(w, h) // 6)
        yy, xx = np.ogrid[:h, :w]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2 <= rad**2).astype(np.uint8)
        segs.append(
            Segment(
                box=Box(cx - rad, cy - rad, cx + rad, cy + rad),
                score=float(rng.uniform(0.6, 0.95)),
                label=labels[i % len(labels)],
                mask=mask,
                class_id=i % len(labels),
            )
        )
    return segs


def _fake_pose(rng: np.random.Generator, w: int, h: int) -> list[PoseInstance]:
    cx, cy = w / 2, h / 2
    keypoints = [
        Keypoint(cx, cy - 60, 0.9, "nose"),
        Keypoint(cx - 20, cy - 40, 0.85, "left_eye"),
        Keypoint(cx + 20, cy - 40, 0.85, "right_eye"),
        Keypoint(cx - 30, cy + 30, 0.7, "left_shoulder"),
        Keypoint(cx + 30, cy + 30, 0.7, "right_shoulder"),
    ]
    return [
        PoseInstance(
            box=Box(cx - 50, cy - 80, cx + 50, cy + 80),
            score=0.92,
            keypoints=keypoints,
        )
    ]


def _fake_obb(rng: np.random.Generator, w: int, h: int) -> list[OrientedDetection]:
    return [
        OrientedDetection(
            box=OrientedBox(
                cx=w / 2, cy=h / 2, w=w / 4, h=h / 5, theta=float(rng.uniform(0, 3.14))
            ),
            score=0.81,
            label="object",
            class_id=0,
        )
    ]


def _factory(entry: ModelEntry) -> MockEngine:
    return MockEngine(entry)


register_engine("mock", _factory)

__all__ = ["MockEngine"]
