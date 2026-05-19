# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.30.1: the public reporting API must be importable from PyPI-installed package.

The v34 notebook imports these names directly from visionservex.reporting.
If any of these fail, the package cannot serve a PyPI-only environment.
"""

from __future__ import annotations


def test_reporting_public_api_exports() -> None:
    """All v2.30+ rendering helpers must be exported from visionservex.reporting."""
    from visionservex.reporting import (
        NOT_APPLICABLE,
        NOT_APPLICABLE_SMOKE,
        NOT_COLLECTED,
        NOT_FOUND,
        NOT_RUN,
        is_nullish,
        render_nullable,
        render_table_for_notebook,
    )

    # Smoke: verify the constants are non-empty strings
    assert isinstance(NOT_APPLICABLE, str) and NOT_APPLICABLE
    assert isinstance(NOT_APPLICABLE_SMOKE, str) and NOT_APPLICABLE_SMOKE
    assert isinstance(NOT_COLLECTED, str) and NOT_COLLECTED
    assert isinstance(NOT_FOUND, str) and NOT_FOUND
    assert isinstance(NOT_RUN, str) and NOT_RUN

    # Smoke: verify the functions are callable
    assert callable(is_nullish)
    assert callable(render_nullable)
    assert callable(render_table_for_notebook)


def test_is_nullish_none() -> None:
    from visionservex.reporting import is_nullish

    assert is_nullish(None) is True
    assert is_nullish(0.0) is False
    assert is_nullish("NaN") is True
    assert is_nullish(float("nan")) is True


def test_render_nullable_no_raw_nan() -> None:
    from visionservex.reporting import render_nullable

    assert "NaN" not in render_nullable(float("nan"))
    assert "NaN" not in render_nullable(None)
    assert render_nullable(0.4567) == "0.4567"


def test_render_table_for_notebook_degrades_gracefully() -> None:
    """render_table_for_notebook must not raise on non-DataFrame input."""
    from visionservex.reporting import render_table_for_notebook

    result = render_table_for_notebook({"key": "value"})
    assert result == {"key": "value"}  # pass-through when pandas not available or non-DF
