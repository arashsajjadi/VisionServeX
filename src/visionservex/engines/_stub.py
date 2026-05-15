# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Shared helper for engine stubs.

Honesty policy (Pass 2):
- Stubs DO NOT silently fall back to MockEngine output.
- A stub only returns mock output when ``settings.models.allow_mock_fallback``
  is true, AND it sets ``fallback_reason`` and a prominent warning on the
  result.
- Otherwise, loading a stubbed model raises :class:`MissingDependencyError`
  with an actionable install hint.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

from visionservex.config import get_settings
from visionservex.engines.base import BaseEngine, MissingDependencyError
from visionservex.engines.mock import MockEngine
from visionservex.registry import ModelEntry


def assert_modules(modules: Iterable[str], *, install_hint: str) -> None:
    """Raise :class:`MissingDependencyError` if any of ``modules`` is absent."""
    missing = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    if missing:
        raise MissingDependencyError(
            f"missing required modules: {', '.join(missing)}",
            install_hint=install_hint,
        )


class StubEngine(BaseEngine):
    """Engine that defers to MockEngine *only* when mock fallback is enabled.

    Subclasses can override ``_real_load``, ``preprocess``, ``infer``,
    ``postprocess`` to wire a real backend. Until they do, the engine is a
    transparent stub.
    """

    real_install_extra: str = ""
    real_modules: tuple[str, ...] = ()
    backend_label: str = "stub"

    def __init__(self, entry: ModelEntry) -> None:
        super().__init__(entry)
        self._real_ready = False
        self._mock = MockEngine(entry)
        self._missing_msg = ""

    def load(self, *, device: str, precision: str) -> None:
        self.device = device
        self.precision = precision
        try:
            if self.real_modules:
                assert_modules(self.real_modules, install_hint=self._install_hint())
            self._real_load(device=device, precision=precision)
            self._real_ready = True
            self._loaded = True
            return
        except MissingDependencyError as exc:
            self._missing_msg = str(exc)
            # Honest path: by default we refuse to load. Only continue if
            # mock fallback is explicitly enabled.
            if not get_settings().models.allow_mock_fallback:
                raise
            self._mock.load(device=device, precision=precision)
            self._real_ready = False
            self._loaded = True

    def unload(self) -> None:
        self._mock.unload()
        super().unload()

    def infer(self, preprocessed: Any, **kwargs: Any) -> Any:
        if self._real_ready:
            raise NotImplementedError(
                f"{self.__class__.__name__}._real_load succeeded but infer() is not implemented"
            )
        return self._mock.infer(preprocessed, **kwargs)

    def postprocess(self, raw: Any, *, image: Any, **kwargs: Any) -> Any:
        if self._real_ready:
            raise NotImplementedError(
                f"{self.__class__.__name__}._real_load succeeded but postprocess() is not implemented"
            )
        result = self._mock.postprocess(raw, image=image, **kwargs)
        result.warnings.append(
            f"{self.__class__.__name__} returned MOCK output (allow_mock_fallback=true). "
            f"Install hint: {self._install_hint()}"
        )
        result.metadata["fallback_reason"] = self._missing_msg or "real backend not wired"
        result.metadata["backend"] = "mock"
        return result

    def _real_load(self, *, device: str, precision: str) -> None:
        raise MissingDependencyError(
            f"{self.__class__.__name__} is not implemented in this build",
            install_hint=self._install_hint(),
        )

    def _install_hint(self) -> str:
        if self.real_install_extra:
            return (
                f"pip install 'visionservex[{self.real_install_extra}]' "
                f"and consult docs/installation.md"
            )
        return "see docs/installation.md"


__all__ = ["StubEngine", "assert_modules"]
