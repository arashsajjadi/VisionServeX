"""Hugging Face Transformers / Timm engine stub."""

from __future__ import annotations

from visionservex.engines._stub import StubEngine
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry


class HuggingFaceEngine(StubEngine):
    real_install_extra = "grounding"
    real_modules = ("transformers",)


def _factory(entry: ModelEntry) -> HuggingFaceEngine:
    return HuggingFaceEngine(entry)


register_engine("huggingface", _factory)

__all__ = ["HuggingFaceEngine"]
