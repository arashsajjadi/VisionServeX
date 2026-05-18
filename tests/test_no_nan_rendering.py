# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.29.0: null metric values must never render as raw 'NaN'."""

from __future__ import annotations

import json
import math
from typing import Any


def render_nullable_metric(value: Any, *, not_available_label: str = "not collected") -> str:
    """Render a nullable metric value as a safe display string.

    Rules:
    - None → not_available_label
    - float('nan') → not_available_label
    - float('inf') → not_available_label
    - Valid number → formatted string
    - String "nan" or "NaN" → not_available_label
    """
    if value is None:
        return not_available_label
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return not_available_label
    if isinstance(value, str) and value.lower() in ("nan", "inf", "-inf", "none", "null"):
        return not_available_label
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_none_renders_not_collected() -> None:
    assert render_nullable_metric(None) == "not collected"


def test_float_nan_renders_not_collected() -> None:
    assert render_nullable_metric(float("nan")) == "not collected"


def test_float_inf_renders_not_collected() -> None:
    assert render_nullable_metric(float("inf")) == "not collected"


def test_string_nan_renders_not_collected() -> None:
    assert render_nullable_metric("NaN") == "not collected"
    assert render_nullable_metric("nan") == "not collected"


def test_valid_float_renders_as_string() -> None:
    rendered = render_nullable_metric(0.4567)
    assert rendered == "0.4567"
    assert "NaN" not in rendered
    assert "nan" not in rendered


def test_zero_renders_correctly() -> None:
    assert render_nullable_metric(0.0) == "0.0000"


def test_custom_label() -> None:
    result = render_nullable_metric(None, not_available_label="not found")
    assert result == "not found"


def test_official_metrics_table_no_nan() -> None:
    """Official metrics table must never emit raw NaN."""
    from visionservex.reporting.official_metrics import build_official_metrics_table

    rows = build_official_metrics_table()
    for row in rows:
        value = row.get("value")
        if value is None:
            continue
        if isinstance(value, float):
            assert not math.isnan(value), (
                f"Official metrics table has raw NaN for {row.get('model_id')}"
            )
        if isinstance(value, str):
            assert value.lower() != "nan", (
                f"Official metrics table has string 'NaN' for {row.get('model_id')}"
            )


def test_json_serialisation_no_nan() -> None:
    """JSON serialisation of metric rows must not produce NaN literals."""
    from visionservex.reporting.official_metrics import build_official_metrics_table

    rows = build_official_metrics_table()
    serialized = json.dumps(rows)
    assert "NaN" not in serialized, "JSON output contains NaN literal"
    assert ": NaN" not in serialized


def test_smoke_matrix_no_nan_in_csv() -> None:
    """Placeholder: smoke matrix CSV must not contain 'NaN' or 'NOT_WIRED' cells.

    This test passes if the smoke-matrix CSV is absent (hasn't been generated yet).
    It becomes a real test once the matrix is produced.
    """
    from pathlib import Path

    csv_paths = [
        Path("reports/model_smoke_matrix_v229.csv"),
        Path("reports/core_smoke_matrix_v229.csv"),
    ]
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        content = csv_path.read_text()
        bad_values = [v for v in ("NaN", "NOT_WIRED", ",nan,", ",NaN,") if v in content]
        assert not bad_values, f"{csv_path}: forbidden values found: {bad_values}"
