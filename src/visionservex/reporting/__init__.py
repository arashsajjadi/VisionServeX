# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.28.0: canonical reporting infrastructure.

This package replaces ad-hoc model-status tables with a single source of
truth. The notebook may only render values produced by:

- :mod:`visionservex.reporting.status_vocab` — allowed final states.
- :mod:`visionservex.reporting.truth_audit` — counts NaN/NOT_WIRED/stale.
- :mod:`visionservex.reporting.state_resolver` — canonical model state.
- :mod:`visionservex.reporting.official_metrics` — null-safe metric table.

Anything outside these modules that builds a final "winner" table is a
v2.28 regression.
"""

from visionservex.reporting.rendering import (
    NOT_APPLICABLE,
    NOT_APPLICABLE_SMOKE,
    NOT_COLLECTED,
    NOT_FOUND,
    NOT_RUN,
    is_nullish,
    render_nullable,
    render_table_for_notebook,
)
from visionservex.reporting.status_vocab import (
    ALLOWED_FINAL_STATES,
    FORBIDDEN_FINAL_STATES,
    STALE_MARKERS,
    WINNER_CONTEXT_STALE_MARKERS,
    legacy_status_to_canonical,
)

__all__ = [
    "ALLOWED_FINAL_STATES",
    "FORBIDDEN_FINAL_STATES",
    "NOT_APPLICABLE",
    "NOT_APPLICABLE_SMOKE",
    "NOT_COLLECTED",
    "NOT_FOUND",
    "NOT_RUN",
    "STALE_MARKERS",
    "WINNER_CONTEXT_STALE_MARKERS",
    "is_nullish",
    "legacy_status_to_canonical",
    "render_nullable",
    "render_table_for_notebook",
]
