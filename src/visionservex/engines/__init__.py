"""Inference engine implementations and registration."""

# ruff: noqa: F401
# All imports below are intentional side-effect imports that register engines
# via register_engine() calls in each module. They appear unused to the linter
# because they are imported only for their side effects.

from visionservex.engines import dfine as _dfine
from visionservex.engines import dinov2 as _dinov2
from visionservex.engines import florence2 as _florence2
from visionservex.engines import grounded_sam as _grounded_sam
from visionservex.engines import grounded_sam2 as _grounded_sam2
from visionservex.engines import grounding_dino as _gd
from visionservex.engines import hf_classify as _hf_classify
from visionservex.engines import huggingface as _hf
from visionservex.engines import libreyolo as _libreyolo  # permissive YOLOX/YOLOv9/RT-DETR/D-FINE
from visionservex.engines import (
    medsam2_sidecar as _medsam2_sidecar,  # research-only MedSAM2 dependency-gated sidecar
)

# Order matters: mock must be first because the stub engines fall back to MockEngine.
from visionservex.engines import mock as _mock
from visionservex.engines import oneformer as _oneformer
from visionservex.engines import onnx as _onnx
from visionservex.engines import openmmlab as _mm
from visionservex.engines import openmmlab_sidecar as _mm_sidecar
from visionservex.engines import owlv2 as _owlv2
from visionservex.engines import pytorch as _torch
from visionservex.engines import rfdetr as _rfdetr
from visionservex.engines import sam2 as _sam
from visionservex.engines import sam2_hf as _sam2_hf
from visionservex.engines import sam_hf as _sam_hf
from visionservex.engines import (
    sam_optional as _sam_optional,  # v2.55: efficientsam, hq-sam, mobilesam
)
from visionservex.engines import swinv2 as _swinv2
from visionservex.engines import (
    torchvision_classification as _tv_classify,  # v3.15: classic permissive classifiers
)
from visionservex.engines.base import BaseEngine, EngineError, MissingDependencyError
from visionservex.engines.mock import MockEngine
from visionservex.engines.registry import build_engine, register_engine

__all__ = [
    "BaseEngine",
    "EngineError",
    "MissingDependencyError",
    "MockEngine",
    "build_engine",
    "register_engine",
]
