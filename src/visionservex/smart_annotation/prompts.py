"""Parse user-supplied prompt inputs (CLI strings / JSON files) into a ``Prompt``."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import Prompt


def parse_box(spec: str | None) -> tuple[float, float, float, float] | None:
    """Parse ``"x1,y1,x2,y2"`` into an xyxy tuple."""
    if not spec:
        return None
    parts = [p.strip() for p in str(spec).replace(" ", "").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(f"--box must be 'x1,y1,x2,y2', got {spec!r}")
    x1, y1, x2, y2 = (float(p) for p in parts)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _coerce_points(obj: Any) -> list[list[float]]:
    """Accept [[x,y],...] or [{'x':..,'y':..},...] or {'points':[...]}."""
    if obj is None:
        return []
    if isinstance(obj, dict):
        obj = obj.get("points", obj.get("polygon", obj.get("polyline", [])))
    pts: list[list[float]] = []
    for p in obj:
        if isinstance(p, dict):
            pts.append([float(p["x"]), float(p["y"])])
        else:
            pts.append([float(p[0]), float(p[1])])
    return pts


def load_points(path: str | Path | None) -> list[list[float]] | None:
    """Load a points JSON file. Returns None if path is falsy."""
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    return _coerce_points(data)


def build_prompt(
    *,
    box: str | None = None,
    polygon: str | Sequence | None = None,
    positive_points: str | Sequence | None = None,
    negative_points: str | Sequence | None = None,
    scribble: str | Sequence | None = None,
    polyline: str | Sequence | None = None,
    mask_hint: np.ndarray | str | Path | None = None,
) -> Prompt:
    """Build a ``Prompt`` from CLI-style inputs (file paths or in-memory seqs)."""

    def _resolve(val: Any) -> list[list[float]] | None:
        if val is None:
            return None
        if isinstance(val, (str, Path)):
            return load_points(val)
        return _coerce_points(val)

    hint: np.ndarray | None = None
    if mask_hint is not None:
        if isinstance(mask_hint, (str, Path)):
            import cv2

            hint = cv2.imread(str(mask_hint), cv2.IMREAD_GRAYSCALE)
            if hint is None:
                raise FileNotFoundError(f"mask_hint not readable: {mask_hint}")
            hint = (hint > 127).astype("uint8")
        else:
            hint = (np.asarray(mask_hint) > 0).astype("uint8")

    return Prompt(
        box=parse_box(box) if isinstance(box, str) else box,
        polygon=_resolve(polygon),
        positive_points=_resolve(positive_points),
        negative_points=_resolve(negative_points),
        scribble=_resolve(scribble),
        polyline=_resolve(polyline),
        mask_hint=hint,
    )
