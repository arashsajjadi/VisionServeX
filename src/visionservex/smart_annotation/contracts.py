"""Prompt + result contracts for the smart-annotation toolkit.

A single, stable schema is used across every classic refiner so the CLI, the
benchmark harness, and downstream consumers never have to special-case a method.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# The output contract advertised in docs/notebooks. Every refiner returns these.
OUTPUT_CONTRACT_KEYS = (
    "mask",  # H x W uint8 {0,1}
    "polygon",  # optional COCO polygon: list[list[float]] (flattened x,y,...)
    "bbox",  # xyxy float
    "confidence",  # optional float in [0,1] or None
    "method",  # str
    "latency_ms",  # float
    "device",  # "cpu"
    "license_safe",  # bool — always True for classic tools (no weights)
)

# Accepted prompt modalities (V3 requirement).
PROMPT_MODALITIES = (
    "box",
    "polygon",
    "positive_points",
    "negative_points",
    "scribble",
    "polyline",
    "mask_hint",
)


@dataclass
class Prompt:
    """A coarse user annotation prompt.

    Coordinates are in pixel space (x, y). All fields are optional; a method
    validates that it received at least one modality it can use.
    """

    box: tuple[float, float, float, float] | None = None  # xyxy
    polygon: Sequence[Sequence[float]] | None = None  # [[x, y], ...]
    positive_points: Sequence[Sequence[float]] | None = None  # [[x, y], ...]
    negative_points: Sequence[Sequence[float]] | None = None  # [[x, y], ...]
    scribble: Sequence[Sequence[float]] | None = None  # foreground scribble pts
    polyline: Sequence[Sequence[float]] | None = None  # ordered contour pts
    mask_hint: np.ndarray | None = None  # H x W bool/uint8 coarse mask

    def is_empty(self) -> bool:
        return all(
            getattr(self, m) is None
            for m in (
                "box",
                "polygon",
                "positive_points",
                "negative_points",
                "scribble",
                "polyline",
                "mask_hint",
            )
        )

    def has(self, modality: str) -> bool:
        return getattr(self, modality, None) is not None


@dataclass
class RefineResult:
    """The canonical refiner output."""

    mask: np.ndarray  # H x W uint8 {0,1}
    bbox: tuple[float, float, float, float]  # xyxy
    method: str
    latency_ms: float
    polygon: list[list[float]] | None = None
    confidence: float | None = None
    device: str = "cpu"
    license_safe: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def area(self) -> int:
        return int(self.mask.astype(bool).sum())

    def to_contract_dict(self, include_mask: bool = False) -> dict[str, Any]:
        """JSON-serialisable view following ``OUTPUT_CONTRACT_KEYS``.

        ``mask`` is omitted by default (large array); ``mask_shape`` / ``mask_area``
        are always included so a consumer can verify a non-empty result without
        carrying the pixels. Set ``include_mask=True`` to embed the array as a list.
        """
        d: dict[str, Any] = {
            "polygon": self.polygon,
            "bbox": [float(v) for v in self.bbox],
            "confidence": self.confidence,
            "method": self.method,
            "latency_ms": round(float(self.latency_ms), 3),
            "device": self.device,
            "license_safe": self.license_safe,
            "mask_shape": list(self.mask.shape),
            "mask_area": self.area,
        }
        if include_mask:
            d["mask"] = self.mask.astype("uint8").tolist()
        return d
