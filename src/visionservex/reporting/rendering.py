# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.30.0: shared null-safe rendering for notebook + CLI tables.

Forbidden: showing raw ``NaN`` to a user. Every CLI/notebook surface that
might display a missing metric must route through ``render_nullable`` so
the user never sees ``"NaN"`` or ``"nan"``.
"""

from __future__ import annotations

import math
from typing import Any

NOT_APPLICABLE = "not applicable"
NOT_COLLECTED = "not collected"
NOT_FOUND = "not found"
NOT_RUN = "not run"
NOT_APPLICABLE_SMOKE = "not applicable (smoke)"

__all__ = [
    "NOT_APPLICABLE",
    "NOT_APPLICABLE_SMOKE",
    "NOT_COLLECTED",
    "NOT_FOUND",
    "NOT_RUN",
    "is_nullish",
    "render_nullable",
    "render_table_for_notebook",
]


def is_nullish(value: Any) -> bool:
    """Return True if ``value`` is None / NaN / inf / 'nan' / empty string."""
    if value is None:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "nan", "inf", "-inf", "none", "null", "na", "n/a"):
            return True
    return False


def render_nullable(
    value: Any,
    *,
    status: str | None = None,
    label: str | None = None,
    decimals: int = 4,
) -> str:
    """Render a nullable metric value as a user-safe string.

    Args:
        value: the raw value (may be None / NaN / float / int / str).
        status: optional context. Recognised values:
            ``not_collected`` → "not collected"
            ``not_found`` → "not found"
            ``not_applicable`` → "not applicable"
            ``not_applicable_smoke`` → "not applicable (smoke)"
            ``dataset_required`` → "dataset required"
            ``smoke_passed`` → if value is also missing, "not applicable (smoke)"
        label: optional override label. If supplied and value is nullish,
            this string is used verbatim instead of any computed label.
        decimals: float formatting precision.

    Rules:
        - Nullish + label given → label.
        - Nullish + status given → status-derived phrase.
        - Nullish, no hints → "not collected".
        - Non-nullish numeric → fixed-decimal string.
        - Non-nullish string → str(value).
    """
    if is_nullish(value):
        if label is not None:
            return label
        if status:
            s = status.lower()
            if s in ("not_collected", "notcollected"):
                return NOT_COLLECTED
            if s in ("not_found", "notfound"):
                return NOT_FOUND
            if s in ("not_applicable", "notapplicable", "n/a", "na"):
                return NOT_APPLICABLE
            if s in ("not_applicable_smoke", "smoke"):
                return NOT_APPLICABLE_SMOKE
            if s in ("dataset_required",):
                return "dataset required"
            if s in ("smoke_passed",):
                return NOT_APPLICABLE_SMOKE
            if s in ("not_run", "notrun"):
                return NOT_RUN
        return NOT_COLLECTED

    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def render_table_for_notebook(df: Any) -> Any:
    """Return a copy of ``df`` with every cell run through ``render_nullable``.

    Numeric columns retain their formatted (string) representation; missing
    cells become user-safe phrases. Operates on a pandas DataFrame if
    available; otherwise returns the input unchanged.

    The function never raises on missing pandas — it simply degrades to a
    pass-through so callers that do not have pandas still work.
    """
    try:
        import pandas as pd
    except ImportError:
        return df

    if not isinstance(df, pd.DataFrame):
        return df

    out = df.copy()
    for col in out.columns:
        # If a known status column exists, use it for context.
        status_col = None
        for candidate in (
            f"{col}_status",
            f"{col}_source_status",
            "source_status",
            "metric_status",
        ):
            if candidate in out.columns and candidate != col:
                status_col = candidate
                break

        rendered = []
        for idx, val in out[col].items():
            status_val: str | None = None
            if status_col is not None:
                raw_status = out.at[idx, status_col]
                if isinstance(raw_status, str):
                    status_val = raw_status
            rendered.append(render_nullable(val, status=status_val))
        out[col] = rendered
    return out
