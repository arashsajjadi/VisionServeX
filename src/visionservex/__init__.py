"""VisionServeX: serve permissive-license computer vision models locally and over Cloudflare Tunnel."""

from __future__ import annotations

__version__ = "3.23.0"
__author__ = "Arash Sajjadi"
__email__ = "arash.sajjadi@usask.ca"
__license__ = "Apache-2.0"

from visionservex.client import AsyncClient, Client, ClientResult, GatewayError
from visionservex.core.model import VisionModel, list_models, model_capabilities
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
    ModelAcknowledgementRequiredError,
    ModelAGPLRestrictedError,
    ModelLicenseError,
    ModelLicenseRestrictedError,
    ModelLicenseReviewRequiredError,
    ModelMissingWeightsError,
    ModelNotCommercialSafeError,
    ModelNotFoundError,
    ModelRequiresBYOCheckpointError,
    ModelUseModeNotAllowedError,
    SidecarNotRunningError,
    VisionServeXError,
)
from visionservex.policy import (
    ACKNOWLEDGEMENT_TEXT,
    ModelLicensePolicy,
    assert_commercial_safe,
    explain_model_license,
    get_model_policy,
    list_commercial_safe_models,
    list_research_models,
)
from visionservex.vsx import VSX, VSXError

__all__ = [
    "ACKNOWLEDGEMENT_TEXT",
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
    "ModelAGPLRestrictedError",
    "ModelAcknowledgementRequiredError",
    "ModelLicenseError",
    "ModelLicensePolicy",
    "ModelLicenseRestrictedError",
    "ModelLicenseReviewRequiredError",
    "ModelMissingWeightsError",
    "ModelNotCommercialSafeError",
    "ModelNotFoundError",
    "ModelRequiresBYOCheckpointError",
    "ModelUseModeNotAllowedError",
    "OpenVocabularyResult",
    "OrientedDetectionResult",
    "PoseResult",
    "SegmentationResult",
    "SidecarNotRunningError",
    "VSXError",
    "VisionModel",
    "VisionServeXError",
    "__version__",
    "assert_commercial_safe",
    "explain_model_license",
    "get_model_policy",
    "list_commercial_safe_models",
    "list_models",
    "list_research_models",
    "model_capabilities",
    "normalize_detection",
    "normalize_detections",
    "parse_api_response",
]
