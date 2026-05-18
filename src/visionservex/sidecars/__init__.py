# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Sidecar package: manage isolated conda environments for SOTA detectors
that cannot share the VisionServeX host environment (DEIMv2, RT-DETRv4, etc.).
"""

from visionservex.sidecars.deimv2_normalize import (
    DEIMV2_CANONICAL_FIELDS,
    normalize_deimv2_output,
)
from visionservex.sidecars.manager import (
    SidecarConfig,
    SidecarExecResult,
    SidecarManager,
    SidecarSpec,
)
from visionservex.sidecars.rtdetrv4_normalize import (
    RTDETRV4_CANONICAL_FIELDS,
    normalize_rtdetrv4_output,
)

__all__ = [
    "DEIMV2_CANONICAL_FIELDS",
    "RTDETRV4_CANONICAL_FIELDS",
    "SidecarConfig",
    "SidecarExecResult",
    "SidecarManager",
    "SidecarSpec",
    "normalize_deimv2_output",
    "normalize_rtdetrv4_output",
]
