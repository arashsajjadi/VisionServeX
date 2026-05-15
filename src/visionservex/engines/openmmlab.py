"""OpenMMLab engine stub.

Real integration requires the OpenMMLab toolchain (mmengine, mmcv, plus the
relevant task package such as mmpose, mmdetection, mmrotate, mmpretrain).
Install via openmim per upstream docs.
"""

from __future__ import annotations

from visionservex.engines._stub import StubEngine
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry


class OpenMMLabEngine(StubEngine):
    real_install_extra = "openmmlab"
    real_modules = ("mmengine",)


def _factory(entry: ModelEntry) -> OpenMMLabEngine:
    return OpenMMLabEngine(entry)


register_engine("openmmlab", _factory)

__all__ = ["OpenMMLabEngine"]
