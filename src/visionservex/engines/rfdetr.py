"""RF-DETR engine stub."""

from __future__ import annotations

from visionservex.engines._stub import StubEngine
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry


class RFDETREngine(StubEngine):
    real_install_extra = "torch"
    real_modules = ("torch",)


def _factory(entry: ModelEntry) -> RFDETREngine:
    return RFDETREngine(entry)


register_engine("rfdetr", _factory)

__all__ = ["RFDETREngine"]
