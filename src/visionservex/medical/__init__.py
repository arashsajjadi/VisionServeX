# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Medical segmentation runtime + training-truth subpackage.

Import-light by design: nothing here imports torch / sam2 / nibabel at module
load time. Heavy upstream stacks (MedSAM2 / SAM2) are imported lazily, only when
a runtime call actually needs them, so the base VisionServeX install stays light.

MedSAM2 weights are RESEARCH/EDUCATION ONLY (non-commercial). Nothing in this
package may label MedSAM2 commercial-safe.
"""

from __future__ import annotations

__all__ = ["medsam2_runtime", "training"]
