"""Engine interface and shared error types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from visionservex.core.results import BaseResult
from visionservex.registry import ModelEntry


class EngineError(RuntimeError):
    """Base class for engine errors."""


class MissingDependencyError(EngineError):
    """Raised when a required optional dependency is not installed."""

    def __init__(self, message: str, *, install_hint: str | None = None) -> None:
        super().__init__(message)
        self.install_hint = install_hint

    def __str__(self) -> str:
        if self.install_hint:
            return f"{super().__str__()}  Install hint: {self.install_hint}"
        return super().__str__()


class BaseEngine(ABC):
    """Abstract engine. All engines implement this interface."""

    def __init__(self, entry: ModelEntry) -> None:
        self.entry = entry
        self.device: str = "cpu"
        self.precision: str = "fp32"
        self._loaded: bool = False

    # ----- lifecycle -----

    @abstractmethod
    def load(self, *, device: str, precision: str) -> None:
        """Load weights to the target device/precision."""

    def unload(self) -> None:
        self._loaded = False

    def warmup(self) -> None:
        """Optional warmup. Default is a no-op."""
        return None

    # ----- inference -----

    def preprocess(self, image: Image.Image, **kwargs: Any) -> Any:
        return image

    @abstractmethod
    def infer(self, preprocessed: Any, **kwargs: Any) -> Any:
        """Run the model and return raw outputs."""

    @abstractmethod
    def postprocess(self, raw: Any, *, image: Image.Image, **kwargs: Any) -> BaseResult:
        """Convert raw outputs into a typed result object."""

    def predict(
        self,
        image: Image.Image,
        *,
        prompts: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> BaseResult:
        pre = self.preprocess(image, prompts=prompts, **kwargs)
        raw = self.infer(pre, prompts=prompts, **kwargs)
        return self.postprocess(raw, image=image, prompts=prompts, **kwargs)

    # ----- capabilities -----

    def supports(self, capability: str) -> bool:
        return capability in {"predict"}

    def export(self, format: str, output_path: Path) -> Path:  # pragma: no cover - default
        raise NotImplementedError(
            f"engine {self.__class__.__name__} does not support export to {format!r}"
        )


__all__ = ["BaseEngine", "EngineError", "MissingDependencyError"]
