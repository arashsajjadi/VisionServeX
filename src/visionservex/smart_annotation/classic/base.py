"""Registry + shared geometry helpers for classic refiners."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from ..contracts import Prompt, RefineResult

# method_name -> (callable, dependency_license_string)
_REGISTRY: dict[str, tuple[Callable[[np.ndarray, Prompt], RefineResult], str]] = {}


def register(name: str, license_str: str):
    """Decorator registering a refiner under ``name`` with its dependency license."""

    def _wrap(fn: Callable[[np.ndarray, Prompt], RefineResult]):
        _REGISTRY[name] = (fn, license_str)
        return fn

    return _wrap


def list_methods() -> list[str]:
    return sorted(_REGISTRY)


# method -> permissive dependency license (no model weights involved)
def license_map() -> dict[str, str]:
    return {name: lic for name, (_, lic) in _REGISTRY.items()}


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def mask_to_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def mask_to_polygon(mask: np.ndarray, max_points: int = 200) -> list[list[float]] | None:
    """Largest external contour as a single flattened COCO polygon [x,y,x,y,...]."""
    import cv2

    m = (mask.astype("uint8") > 0).astype("uint8")
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) <= 0:
        return None
    # Optionally simplify to keep polygons compact.
    eps = 0.001 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    pts = approx.reshape(-1, 2)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    return [[float(x), float(y)] for x, y in pts]


def box_from_prompt(prompt: Prompt, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Resolve an integer xyxy box from whatever the prompt supplies."""
    h, w = shape[:2]
    if prompt.box is not None:
        x1, y1, x2, y2 = prompt.box
    else:
        pts: list[list[float]] = []
        for src in (prompt.polygon, prompt.positive_points, prompt.scribble, prompt.polyline):
            if src is not None:
                pts.extend([list(p) for p in src])
        if prompt.mask_hint is not None and prompt.mask_hint.any():
            return tuple(int(v) for v in mask_to_bbox(prompt.mask_hint))  # type: ignore[return-value]
        if not pts:
            raise ValueError("prompt has no box/points/polygon/mask_hint to localise")
        arr = np.asarray(pts, dtype=float)
        # pad a small margin around bare points
        pad = max(4.0, 0.05 * max(h, w))
        x1, y1 = arr[:, 0].min() - pad, arr[:, 1].min() - pad
        x2, y2 = arr[:, 0].max() + pad, arr[:, 1].max() + pad
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, x1 + 1, w))
    y2 = int(np.clip(y2, y1 + 1, h))
    return (x1, y1, x2, y2)


def rasterize_points(points, shape: tuple[int, int], radius: int = 4) -> np.ndarray:
    """Disk-stamp a set of points into a uint8 seed image."""
    import cv2

    seed = np.zeros(shape[:2], dtype="uint8")
    if points is None:
        return seed
    for x, y in points:
        cv2.circle(seed, (round(float(x)), round(float(y))), radius, 1, -1)
    return seed


def finalize(
    mask: np.ndarray, method: str, t0: float, confidence: float | None = None, **meta
) -> RefineResult:
    """Common post-processing: clean, bbox, polygon, timing → RefineResult."""
    m = (np.asarray(mask) > 0).astype("uint8")
    return RefineResult(
        mask=m,
        bbox=mask_to_bbox(m),
        method=method,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
        polygon=mask_to_polygon(m),
        confidence=confidence,
        device="cpu",
        license_safe=True,
        meta=meta,
    )
