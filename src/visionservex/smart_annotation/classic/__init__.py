"""Classic (weight-free) interactive refiners + dispatch."""

from __future__ import annotations

import numpy as np

from ..contracts import Prompt, RefineResult
from . import methods as _methods  # noqa: F401  (import registers the methods)
from .base import _REGISTRY, license_map, list_methods

METHOD_LICENSE = license_map()

__all__ = ["METHOD_LICENSE", "list_methods", "refine"]


def refine(image: np.ndarray, prompt: Prompt, method: str = "classic-grabcut") -> RefineResult:
    """Run a classic refiner. ``image`` is HxWx3 (BGR/uint8) or HxW grayscale."""
    if method not in _REGISTRY:
        raise KeyError(f"unknown classic method {method!r}; available: {', '.join(list_methods())}")
    if prompt is None or prompt.is_empty():
        raise ValueError("prompt is empty; supply box/points/polygon/scribble/mask_hint")
    fn, _lic = _REGISTRY[method]
    return fn(image, prompt)
