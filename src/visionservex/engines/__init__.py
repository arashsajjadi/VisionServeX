"""Inference engine implementations and registration."""

from visionservex.engines.base import BaseEngine, EngineError, MissingDependencyError
from visionservex.engines.registry import build_engine, register_engine

# Order matters: mock first because the stub engines fall back to MockEngine.
from visionservex.engines import mock as _mock  # noqa: F401,E402
from visionservex.engines import (  # noqa: F401,E402
    dfine as _dfine,
    grounding_dino as _gd,
    huggingface as _hf,
    onnx as _onnx,
    openmmlab as _mm,
    pytorch as _torch,
    rfdetr as _rfdetr,
    sam2 as _sam,
)
from visionservex.engines.mock import MockEngine  # noqa: E402

__all__ = [
    "BaseEngine",
    "EngineError",
    "MissingDependencyError",
    "MockEngine",
    "build_engine",
    "register_engine",
]
