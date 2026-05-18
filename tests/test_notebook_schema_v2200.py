# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Phase 1/2 (v2.20.0 / notebook v23): schema-normalization utility regression test.

The v22 notebook crashed with `KeyError: 'model'` in section 12 because
`segmentation_rows = []` produced a column-less DataFrame, and downstream
code (cells 30/32/34) did `df["model"]` / `sort_values(["mAP50_95",
"AP50"])` without a schema guarantee.

v23 introduced a global schema-normalization utility cell that runs
right after the helpers cell. These tests extract that utility from the
notebook JSON, execute it in a fresh namespace, and assert it survives
the exact failure shapes the v22 crash exhibited.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

NB_PATH = Path(
    "/home/arash/PycharmProjects/VisionServeX/notebook/"
    "VisionServeX_Colab_Universal_Model_Audit_Benchmark.ipynb"
)


def _load_utility_namespace() -> dict:
    """Find the v23 utility cell, exec it, return its namespace."""
    nb = json.loads(NB_PATH.read_text())
    util_src = None
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"])
        if "v23 SCHEMA NORMALIZATION UTILITY" in src:
            util_src = src
            break
    if util_src is None:
        pytest.skip("v23 utility cell not present in notebook")
    ns: dict = {"__builtins__": __builtins__}
    exec(util_src, ns, ns)
    return ns


# ---------------------------------------------------------------------------
# ensure_columns
# ---------------------------------------------------------------------------


def test_ensure_columns_on_empty_df() -> None:
    ns = _load_utility_namespace()
    df = ns["ensure_columns"](
        pd.DataFrame(), defaults={"model": "", "status": "unknown", "AP50": float("nan")}
    )
    assert "model" in df.columns
    assert "status" in df.columns
    assert "AP50" in df.columns
    assert len(df) == 0


def test_ensure_columns_on_partial_df() -> None:
    ns = _load_utility_namespace()
    src = pd.DataFrame([{"model": "dfine-s", "source": "VSX"}])
    df = ns["ensure_columns"](
        src, defaults={"model": "", "status": "unknown", "AP50": float("nan")}
    )
    # Existing values preserved
    assert df.iloc[0]["model"] == "dfine-s"
    # Missing columns filled
    assert df.iloc[0]["status"] == "unknown"
    assert pd.isna(df.iloc[0]["AP50"])


# ---------------------------------------------------------------------------
# normalize_common_result_schema
# ---------------------------------------------------------------------------


def test_normalize_segmentation_kind_adds_n_masks() -> None:
    ns = _load_utility_namespace()
    df = ns["normalize_common_result_schema"](
        pd.DataFrame([{"model": "sam2-tiny"}]),
        section_name="12_seg",
        kind="segmentation",
    )
    assert "n_masks" in df.columns
    assert "mask_metric_valid" in df.columns
    assert df.iloc[0]["section"] == "12_seg"
    # benchmark_or_smoke defaults to visual_smoke for segmentation kind
    assert df.iloc[0]["benchmark_or_smoke"] == "visual_smoke"


def test_normalize_detection_kind_adds_metric_columns() -> None:
    ns = _load_utility_namespace()
    df = ns["normalize_common_result_schema"](
        pd.DataFrame([{"model": "dfine-s-o365-coco"}]),
        section_name="9_det",
        kind="detection",
    )
    for col in ("AP50", "AP75", "mAP50_95", "class_agnostic_AP50", "latency_ms_p50"):
        assert col in df.columns


def test_normalize_on_completely_empty_df_returns_schema_only() -> None:
    """The exact v22 crash shape: segmentation_rows = [] -> empty df, no columns."""
    ns = _load_utility_namespace()
    df = ns["normalize_common_result_schema"](
        pd.DataFrame(), section_name="12_seg", kind="segmentation"
    )
    # All common columns present even on empty df.
    for col in ("model", "source", "section", "status", "blocker_code"):
        assert col in df.columns, col
    # Downstream code can now safely do df["model"] (returns empty Series).
    s = df["model"]
    assert len(s) == 0


# ---------------------------------------------------------------------------
# safe_str_series
# ---------------------------------------------------------------------------


def test_safe_str_series_on_missing_column_returns_empty() -> None:
    ns = _load_utility_namespace()
    s = ns["safe_str_series"](pd.DataFrame([{"x": 1}]), "model")
    assert len(s) == 1
    assert s.iloc[0] == ""


def test_safe_str_series_on_present_column_returns_strings() -> None:
    ns = _load_utility_namespace()
    s = ns["safe_str_series"](pd.DataFrame([{"model": "foo"}]), "model")
    assert s.iloc[0] == "foo"


# ---------------------------------------------------------------------------
# ensure_safe_for_plot
# ---------------------------------------------------------------------------


def test_ensure_safe_for_plot_on_empty_df() -> None:
    """Cell 32 calls plot_df['model'] inside `if not plot_df.empty:`.
    The bug: when df_det_all had rows but dropna emptied plot_df, the
    `pd.DataFrame()` fallback returned a column-less df; section 32's
    `~plot_df["model"]` crashed. ensure_safe_for_plot guarantees the
    columns even on the empty-fallback path.
    """
    ns = _load_utility_namespace()
    df = ns["ensure_safe_for_plot"](pd.DataFrame())
    assert "model" in df.columns
    assert "latency_ms_p50" in df.columns
    assert "AP50" in df.columns
    assert "mAP50_95" in df.columns


def test_section_12_failure_shape_does_not_crash() -> None:
    """Reproduce the exact v22 failure shape: empty segmentation_rows + downstream df['model'] access."""
    ns = _load_utility_namespace()
    segmentation_rows = []
    df_seg = ns["normalize_common_result_schema"](
        pd.DataFrame(segmentation_rows),
        section_name="12_segmentation",
        kind="segmentation",
    )
    # Downstream operations must not raise.
    _ = df_seg["model"]
    _ = df_seg.get("model")
    _ = df_seg.empty
    assert df_seg.empty
    assert "model" in df_seg.columns


def test_v23_schema_utility_marker_is_present_in_notebook() -> None:
    """The notebook must keep the v23 schema utility marker across later releases."""
    nb = json.loads(NB_PATH.read_text())
    text = " ".join("".join(c["source"]) for c in nb["cells"])
    # Schema utility cell is the durable artifact; it persists across v23/v24/...
    assert "v23 SCHEMA NORMALIZATION UTILITY" in text
    # Version markers move with each release; just assert they exist and are >= 2.20.
    import re

    ver_match = re.search(r'VISION_SERVEX_VERSION\s*=\s*"(\d+)\.(\d+)\.(\d+)"', text)
    assert ver_match, "VISION_SERVEX_VERSION constant missing"
    major, minor, _patch = (int(x) for x in ver_match.groups())
    assert (major, minor) >= (2, 20), (major, minor)
    nb_match = re.search(r'NOTEBOOK_VERSION\s*=\s*"v(\d+)"', text)
    assert nb_match, "NOTEBOOK_VERSION constant missing"
    assert int(nb_match.group(1)) >= 23
