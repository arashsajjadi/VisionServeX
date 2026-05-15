"""Generic PyTorch engine stub.

A real PyTorch engine would build the right model architecture for the given
registry entry, load weights, and return typed results. Until that is wired
up per-family, this stub falls back to MockEngine output and exposes a clear
install hint when the user wants the real backend.
"""

from __future__ import annotations

from visionservex.engines._stub import StubEngine
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry


class PyTorchEngine(StubEngine):
    real_install_extra = "torch"
    real_modules = ("torch",)


def _factory(entry: ModelEntry) -> PyTorchEngine:
    return PyTorchEngine(entry)


register_engine("pytorch", _factory)

__all__ = ["PyTorchEngine"]
