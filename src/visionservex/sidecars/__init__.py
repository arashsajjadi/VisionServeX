# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Sidecar package: manage isolated conda environments for SOTA detectors
that cannot share the VisionServeX host environment (DEIMv2, RT-DETRv4, etc.).
"""

from visionservex.sidecars.manager import (
    SidecarConfig,
    SidecarExecResult,
    SidecarManager,
    SidecarSpec,
)

__all__ = [
    "SidecarConfig",
    "SidecarExecResult",
    "SidecarManager",
    "SidecarSpec",
]
