# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7: RF-DETR-Seg variants — API, states, real execution evidence."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

R = Path(__file__).parent.parent / "notebook" / "99_final_report" / "reports"
ALL = [
    "rfdetr-seg-nano",
    "rfdetr-seg-small",
    "rfdetr-seg-medium",
    "rfdetr-seg-large",
    "rfdetr-seg-xl",
    "rfdetr-seg-2xl",
]


def test_runtime_module_has_all_six():
    from visionservex.rfdetr_seg_runtime import variants

    assert set(variants()) == set(ALL)


@pytest.mark.parametrize("m", ALL)
def test_explain_apache_commercial_safe(m):
    from visionservex.rfdetr_seg_runtime import explain

    info = explain(m)
    assert info["license"] == "Apache-2.0"
    assert info["commercial_safe"] is True
    assert info["state"] == "benchmark_passed"


def test_vsx_segment_instances_routing():
    from visionservex.vsx import VSX, VSXError

    h = VSX.rfdetr_seg("rfdetr-seg-small")
    assert h.explain()["state"] == "benchmark_passed"
    # non-instance-seg model on VSX(...).segment_instances must raise structured
    with pytest.raises(VSXError):
        VSX("sam-vit-base").segment_instances("x.jpg")


def test_three_plus_executed_with_masks():
    led = list(csv.DictReader((R / "v37_new_model_execution_ledger.csv").open()))
    ok = [r for r in led if r["task"].startswith("rfdetrseg:") and r["status"] == "ok"]
    assert len(ok) >= 3
    for _r in ok:
        assert Path(R.parent / "artifacts" / "v37").exists()


def test_seg_xl_2xl_not_pml():
    """SEG XL/2XL are Apache and must NOT be excluded as PML."""
    from visionservex.rfdetr_seg_runtime import explain

    for m in ["rfdetr-seg-xl", "rfdetr-seg-2xl"]:
        info = explain(m)
        assert "PML" not in info["license"]
        assert info["commercial_safe"] is True
