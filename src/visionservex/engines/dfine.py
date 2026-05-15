"""D-FINE engine stub."""

from __future__ import annotations

from visionservex.engines._stub import StubEngine
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry


class DFINEEngine(StubEngine):
    real_install_extra = "torch"
    real_modules = ("torch",)


def _factory(entry: ModelEntry) -> DFINEEngine:
    return DFINEEngine(entry)


register_engine("dfine", _factory)

__all__ = ["DFINEEngine"]
