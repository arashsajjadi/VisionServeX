"""ONNX Runtime engine stub."""

from __future__ import annotations

from visionservex.engines._stub import StubEngine
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry


class ONNXRuntimeEngine(StubEngine):
    real_install_extra = "onnx"
    real_modules = ("onnxruntime",)


def _factory(entry: ModelEntry) -> ONNXRuntimeEngine:
    return ONNXRuntimeEngine(entry)


register_engine("onnx", _factory)

__all__ = ["ONNXRuntimeEngine"]
