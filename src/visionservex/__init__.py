"""VisionServeX: serve permissive-license computer vision models locally and over Cloudflare Tunnel."""

from __future__ import annotations

__version__ = "0.5.0"
__author__ = "Arash Sajjadi"
__email__ = "arash.sajjadi@usask.ca"
__license__ = "Apache-2.0"

from visionservex.core.model import VisionModel
from visionservex.core.results import (
    ClassificationResult,
    DetectionResult,
    OpenVocabularyResult,
    OrientedDetectionResult,
    PoseResult,
    SegmentationResult,
)

__all__ = [
    "VisionModel",
    "DetectionResult",
    "SegmentationResult",
    "PoseResult",
    "ClassificationResult",
    "OrientedDetectionResult",
    "OpenVocabularyResult",
    "__version__",
]
