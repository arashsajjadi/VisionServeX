"""SAM 2 / SAM 2.1 engine stub.

Real integration uses the upstream sam2 package and checkpoints from
https://github.com/facebookresearch/sam2.
"""

from __future__ import annotations

from visionservex.engines._stub import StubEngine
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry


class SAM2Engine(StubEngine):
    real_install_extra = "sam2"
    real_modules = ("torch",)  # sam2 package itself is installed from upstream


def _factory(entry: ModelEntry) -> SAM2Engine:
    return SAM2Engine(entry)


register_engine("sam2", _factory)

__all__ = ["SAM2Engine"]
