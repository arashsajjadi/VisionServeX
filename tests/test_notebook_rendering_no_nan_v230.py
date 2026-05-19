# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.30.0: render_nullable / render_table_for_notebook must never emit raw NaN."""

from __future__ import annotations

from typing import Any


def test_render_nullable_none() -> None:
    from visionservex.reporting import render_nullable

    assert render_nullable(None) == "not collected"


def test_render_nullable_float_nan() -> None:
    from visionservex.reporting import render_nullable

    out = render_nullable(float("nan"))
    assert "NaN" not in out
    assert "nan" not in out
    assert out == "not collected"


def test_render_nullable_status_overrides() -> None:
    from visionservex.reporting import render_nullable

    assert render_nullable(None, status="not_collected") == "not collected"
    assert render_nullable(None, status="not_found") == "not found"
    assert render_nullable(None, status="not_applicable") == "not applicable"
    assert render_nullable(None, status="dataset_required") == "dataset required"
    assert render_nullable(None, status="smoke_passed") == "not applicable (smoke)"


def test_render_nullable_valid_float() -> None:
    from visionservex.reporting import render_nullable

    out = render_nullable(0.4567)
    assert out == "0.4567"


def test_render_nullable_zero() -> None:
    from visionservex.reporting import render_nullable

    assert render_nullable(0.0) == "0.0000"


def test_render_nullable_int_and_bool() -> None:
    from visionservex.reporting import render_nullable

    assert render_nullable(42) == "42"
    assert render_nullable(True) == "yes"
    assert render_nullable(False) == "no"


def test_render_nullable_string_nan() -> None:
    from visionservex.reporting import render_nullable

    for s in ("NaN", "nan", "None", "null", "", "  "):
        out = render_nullable(s)
        assert "NaN" not in out
        assert "nan" not in out


def test_render_table_returns_pandas_dataframe() -> None:
    try:
        import pandas as pd
    except ImportError:
        return  # pandas optional

    from visionservex.reporting import render_table_for_notebook

    df = pd.DataFrame(
        {
            "mAP": [0.45, float("nan"), 0.30],
            "label": ["a", "b", "c"],
        }
    )
    rendered = render_table_for_notebook(df)
    text = rendered.to_string()
    assert "NaN" not in text
    assert "nan" not in text


def test_render_table_pass_through_non_df() -> None:
    from visionservex.reporting import render_table_for_notebook

    obj: Any = {"foo": "bar"}
    assert render_table_for_notebook(obj) is obj


def test_render_table_handles_status_column() -> None:
    try:
        import pandas as pd
    except ImportError:
        return

    from visionservex.reporting import render_table_for_notebook

    df = pd.DataFrame(
        {
            "value": [None, 0.5],
            "value_status": ["not_collected", "ok"],
        }
    )
    rendered = render_table_for_notebook(df)
    text = rendered.to_string()
    assert "not collected" in text
