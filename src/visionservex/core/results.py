"""Stable result schemas returned by every engine.

The shapes here are part of the public API contract. Engines must produce
these objects; never raw dictionaries. Tests assert on their JSON output.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -------------------- primitives --------------------


@dataclass(frozen=True)
class Box:
    """Axis-aligned bounding box in pixel coordinates (xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class OrientedBox:
    """Oriented bounding box: center, size, rotation in radians (CCW)."""

    cx: float
    cy: float
    w: float
    h: float
    theta: float

    def corners(self) -> list[tuple[float, float]]:
        cos = np.cos(self.theta)
        sin = np.sin(self.theta)
        hw, hh = self.w / 2, self.h / 2
        corners = []
        for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
            x = self.cx + dx * cos - dy * sin
            y = self.cy + dx * sin + dy * cos
            corners.append((float(x), float(y)))
        return corners


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    score: float
    name: str | None = None


@dataclass(kw_only=True)
class Detection:
    box: Box
    score: float
    label: str
    class_id: int | None = None


@dataclass(kw_only=True)
class OrientedDetection:
    box: OrientedBox
    score: float
    label: str
    class_id: int | None = None


@dataclass(kw_only=True)
class Segment:
    """A single instance segmentation result.

    ``mask`` is a 2D uint8 numpy array (H, W) with values 0 or 1. ``box`` is
    the tight bounding box of the mask in pixel coordinates.
    """

    box: Box
    score: float
    label: str
    mask: np.ndarray
    class_id: int | None = None


@dataclass(kw_only=True)
class PoseInstance:
    box: Box | None
    score: float
    keypoints: list[Keypoint]


# -------------------- base result --------------------

ResultKind = Literal[
    "detection",
    "segmentation",
    "pose",
    "classification",
    "obb",
    "open_vocab",
]


@dataclass(kw_only=True)
class BaseResult:
    """Common fields for every result returned by ``VisionModel``."""

    kind: ResultKind = "detection"
    model_id: str = ""
    task: str = ""
    image_size: tuple[int, int] = (0, 0)  # (width, height) of source image
    device: str = "cpu"
    precision: str = "fp32"
    backend: str = ""
    model_loaded_from: str | None = None  # cache | huggingface | github | etc.
    cache_path: str | None = None
    latency_ms: float = 0.0
    fallback_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    _image: Image.Image | None = field(default=None, repr=False)

    # ----- serialization -----

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary describing this result."""
        return _result_to_dict(self)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    def to_coco(self) -> dict[str, Any]:  # pragma: no cover - default raises
        raise NotImplementedError(f"to_coco() is not defined for {self.__class__.__name__}")

    # ----- visualization -----

    def plot(
        self,
        image: Image.Image | None = None,
        *,
        font_size: int = 14,
    ) -> Image.Image:
        """Render annotations onto a copy of the image.

        If no image is provided, falls back to ``self._image`` if known.
        """
        src = image or self._image
        if src is None:
            raise ValueError("plot() requires an image; pass one or set source via VisionModel")
        canvas = src.copy().convert("RGB")
        _draw_overlay(canvas, self, font_size=font_size)
        return canvas

    def save(self, path: str | Path, *, format: str | None = None) -> Path:
        """Save the result.

        - ``.json`` writes the result as JSON.
        - any other extension renders ``plot()`` and saves the image.
        """
        out = Path(path)
        suffix = out.suffix.lower().lstrip(".")
        if suffix == "json" or format == "json":
            out.write_text(self.to_json(indent=2), encoding="utf-8")
            return out
        annotated = self.plot()
        annotated.save(out, format=format)
        return out

    def save_json(self, path: str | Path) -> Path:
        """Save result as JSON (alias for ``save(path.json)``)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(indent=2), encoding="utf-8")
        return out

    def save_image(self, path: str | Path) -> Path:
        """Save annotated image (alias for ``save(path.jpg)``)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        annotated = self.plot()
        annotated.save(out)
        return out

    def summary(self) -> str:
        """Return a short, human-readable single-line summary."""
        return f"<{self.__class__.__name__} model={self.model_id} latency={self.latency_ms:.1f}ms>"

    def __str__(self) -> str:
        return self.summary()


# -------------------- concrete result classes --------------------


@dataclass(kw_only=True)
class DetectionResult(BaseResult):
    kind: ResultKind = "detection"
    detections: list[Detection] = field(default_factory=list)

    def to_coco(self) -> dict[str, Any]:
        return {
            "categories": _coco_categories(d.label for d in self.detections),
            "annotations": [
                {
                    "bbox": list(d.box.to_xywh()),
                    "score": d.score,
                    "category_id": d.class_id if d.class_id is not None else 0,
                    "label": d.label,
                }
                for d in self.detections
            ],
        }

    def summary(self) -> str:
        return (
            f"<DetectionResult n={len(self.detections)} model={self.model_id} "
            f"device={self.device} latency={self.latency_ms:.1f}ms>"
        )


@dataclass(kw_only=True)
class SegmentationResult(BaseResult):
    kind: ResultKind = "segmentation"
    segments: list[Segment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        # masks are large and not JSON-serialisable; emit shape + RLE-like length.
        base = _result_to_dict(self)
        for raw, payload in zip(self.segments, base["segments"], strict=True):
            mask = raw.mask
            payload.pop("mask", None)
            payload["mask_shape"] = list(mask.shape)
            payload["mask_pixels_on"] = int(np.count_nonzero(mask))
        return base

    def to_coco(self) -> dict[str, Any]:
        return {
            "categories": _coco_categories(s.label for s in self.segments),
            "annotations": [
                {
                    "bbox": list(s.box.to_xywh()),
                    "score": s.score,
                    "category_id": s.class_id if s.class_id is not None else 0,
                    "label": s.label,
                    "segmentation_shape": list(s.mask.shape),
                }
                for s in self.segments
            ],
        }


@dataclass(kw_only=True)
class PoseResult(BaseResult):
    kind: ResultKind = "pose"
    persons: list[PoseInstance] = field(default_factory=list)


@dataclass(kw_only=True)
class ClassificationResult(BaseResult):
    kind: ResultKind = "classification"
    top_k: list[tuple[str, float]] = field(default_factory=list)  # (label, score)


@dataclass(kw_only=True)
class OrientedDetectionResult(BaseResult):
    kind: ResultKind = "obb"
    detections: list[OrientedDetection] = field(default_factory=list)


@dataclass(kw_only=True)
class OpenVocabularyResult(BaseResult):
    """Result of an open-vocabulary / text-prompted detector."""

    kind: ResultKind = "open_vocab"
    prompts: list[str] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)


# -------------------- helpers --------------------


def _result_to_dict(result: BaseResult) -> dict[str, Any]:
    data = asdict(result)
    data.pop("_image", None)
    return data


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON-serializable")


def _coco_categories(labels: Iterable[str]) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    for lab in labels:
        if lab not in seen:
            seen[lab] = len(seen)
    return [{"id": idx, "name": name} for name, idx in seen.items()]


def _draw_overlay(image: Image.Image, result: BaseResult, *, font_size: int) -> None:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(font_size)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()

    if isinstance(result, (DetectionResult, OpenVocabularyResult)):
        for det in result.detections:
            _draw_box(draw, det.box, f"{det.label} {det.score:.2f}", font)
    elif isinstance(result, SegmentationResult):
        for seg in result.segments:
            _draw_box(draw, seg.box, f"{seg.label} {seg.score:.2f}", font)
            _overlay_mask(image, seg.mask)
    elif isinstance(result, PoseResult):
        for person in result.persons:
            if person.box:
                _draw_box(draw, person.box, f"person {person.score:.2f}", font)
            for kp in person.keypoints:
                draw.ellipse(
                    [kp.x - 3, kp.y - 3, kp.x + 3, kp.y + 3],
                    fill=(255, 255, 0),
                )
    elif isinstance(result, OrientedDetectionResult):
        for det in result.detections:
            corners = det.box.corners()
            draw.line([*corners, corners[0]], fill=(255, 0, 0), width=2)
            draw.text(corners[0], f"{det.label} {det.score:.2f}", fill=(255, 0, 0), font=font)
    elif isinstance(result, ClassificationResult):
        text = " | ".join(f"{lbl}:{score:.2f}" for lbl, score in result.top_k[:3])
        draw.text((10, 10), text, fill=(255, 255, 0), font=font)


def _draw_box(draw: ImageDraw.ImageDraw, box: Box, label: str, font: ImageFont.ImageFont) -> None:
    draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=(0, 255, 0), width=2)
    draw.text((box.x1, max(0, box.y1 - 12)), label, fill=(0, 255, 0), font=font)


def _overlay_mask(image: Image.Image, mask: np.ndarray) -> None:
    if mask.size == 0:
        return
    arr = np.array(image, dtype=np.uint8)
    h, w = arr.shape[:2]
    m = mask
    if m.shape != (h, w):
        m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h)))
        m = (m > 127).astype(np.uint8)
    color = np.array([255, 0, 255], dtype=np.uint8)
    alpha = 0.4
    mask_bool = m.astype(bool)
    arr[mask_bool] = (alpha * color + (1 - alpha) * arr[mask_bool]).astype(np.uint8)
    image.paste(Image.fromarray(arr))


__all__ = [
    "BaseResult",
    "Box",
    "ClassificationResult",
    "Detection",
    "DetectionResult",
    "Keypoint",
    "OpenVocabularyResult",
    "OrientedBox",
    "OrientedDetection",
    "OrientedDetectionResult",
    "PoseInstance",
    "PoseResult",
    "Segment",
    "SegmentationResult",
]
