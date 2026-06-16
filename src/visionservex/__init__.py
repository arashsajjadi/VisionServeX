"""VisionServeX: serve permissive-license computer vision models locally and over Cloudflare Tunnel."""

from __future__ import annotations

__version__ = "3.14.0"
__author__ = "Arash Sajjadi"
__email__ = "arash.sajjadi@usask.ca"
__license__ = "Apache-2.0"

from visionservex.client import AsyncClient, Client, ClientResult, GatewayError
from visionservex.core.model import VisionModel
from visionservex.core.normalizer import (
    normalize_detection,
    normalize_detections,
    parse_api_response,
)
from visionservex.core.results import (
    ClassificationResult,
    DetectionResult,
    OpenVocabularyResult,
    OrientedDetectionResult,
    PoseResult,
    SegmentationResult,
)
from visionservex.exceptions import (
    DeviceUnavailableError,
    EngineDependencyError,
    ExternalModelError,
    InputNotFoundError,
    ManualModelError,
    ModelMissingWeightsError,
    ModelNotFoundError,
    SidecarNotRunningError,
    VisionServeXError,
)
from visionservex.vsx import VSX, VSXError

__all__ = [
    "VSX",
    "AsyncClient",
    "ClassificationResult",
    "Client",
    "ClientResult",
    "DetectionResult",
    "DeviceUnavailableError",
    "EngineDependencyError",
    "ExternalModelError",
    "GatewayError",
    "InputNotFoundError",
    "ManualModelError",
    "ModelMissingWeightsError",
    "ModelNotFoundError",
    "OpenVocabularyResult",
    "OrientedDetectionResult",
    "PoseResult",
    "SegmentationResult",
    "SidecarNotRunningError",
    "VSXError",
    "VisionModel",
    "VisionServeXError",
    "__version__",
    "normalize_detection",
    "normalize_detections",
    "parse_api_response",
]
