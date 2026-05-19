"""Shared display helpers. Never show NaN, NOT_WIRED, or failed_runtime."""

from __future__ import annotations

from visionservex.reporting import render_nullable


def clean(v, *, status=None):
    return render_nullable(v, status=status)


def clean_df(df):
    from visionservex.reporting import render_table_for_notebook

    return render_table_for_notebook(df)


FORBIDDEN = ("NOT_WIRED", "NaN", "v20:", "v2.16", "UNAVAILABLE_OR_FAILED")


def scan_text(text: str) -> list[str]:
    hits = []
    for needle in FORBIDDEN:
        if needle in text:
            hits.append(needle)
    return hits
