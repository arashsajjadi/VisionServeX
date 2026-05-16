# SPDX-License-Identifier: Apache-2.0
"""Readiness metrics package.

A factor's readiness is the pair ``(functional_readiness,
operational_readiness)``. A row is "release-ready" when

    functional_readiness >= 90
    OR (operational_readiness >= 90 AND blocker_certainty >= 95).

This split is what allows VisionServeX to call a row 100% ready even
when the underlying model cannot ship a runnable upstream — the
operational side covers "we know exactly why, we know exactly how a user
can unblock it, and we ship the script that does it."
"""

from visionservex.readiness.metrics import (
    READINESS_ROWS,
    ReadinessRow,
    compute_readiness_table,
    is_row_release_ready,
)

__all__ = [
    "READINESS_ROWS",
    "ReadinessRow",
    "compute_readiness_table",
    "is_row_release_ready",
]
