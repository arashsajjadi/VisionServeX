"""Engine factory.

Engines register themselves by string name. ``build_engine`` instantiates the
engine for a registry entry. Unknown or unavailable backends raise a
:class:`MissingDependencyError` with actionable install hints.
"""

from __future__ import annotations

from collections.abc import Callable

from visionservex.engines.base import BaseEngine, EngineError
from visionservex.registry import ModelEntry

_EngineFactory = Callable[[ModelEntry], BaseEngine]
_FACTORIES: dict[str, _EngineFactory] = {}


def register_engine(name: str, factory: _EngineFactory) -> None:
    """Register an engine factory by name."""
    _FACTORIES[name] = factory


def build_engine(entry: ModelEntry) -> BaseEngine:
    """Instantiate the engine declared by ``entry.engine``."""
    try:
        factory = _FACTORIES[entry.engine]
    except KeyError as exc:
        raise EngineError(
            f"no engine registered for {entry.engine!r}. Available engines: {sorted(_FACTORIES)}"
        ) from exc
    return factory(entry)


__all__ = ["build_engine", "register_engine"]
