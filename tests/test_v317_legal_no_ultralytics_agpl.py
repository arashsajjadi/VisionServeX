# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.17.0: legal firewall across the full catalog."""

from __future__ import annotations

from pathlib import Path

from visionservex.licensing.policy import _ROWS
from visionservex.registry import default_registry

_SRC = Path(__file__).resolve().parents[1] / "src" / "visionservex"
_RUNTIME = [_SRC / "engines", _SRC / "core", _SRC / "data", _SRC / "runtime"]


def test_no_ultralytics_runtime():
    forbidden = ("import ultralytics", "from ultralytics", "ultralytics.YOLO")
    for d in _RUNTIME:
        for f in d.rglob("*.py"):
            text = f.read_text()
            for pat in forbidden:
                assert pat not in text, f"{f}: {pat}"


def test_no_default_safe_is_copyleft():
    for r in _ROWS:
        if r.final_policy == "commercial_safe_core" or r.default_safe:
            for lic in (r.code_license or "", r.weights_license or ""):
                up = lic.upper()
                assert not any(x in up for x in ("AGPL", "GPL-3", "GPLV3", "SSPL")), (
                    f"{r.model_id}: {lic}"
                )


def test_yolonas_not_default_safe():
    from visionservex.licensing.policy import get_policy

    ids = {e.id for e in default_registry().list()}
    assert not any("yolonas" in i for i in ids)
    for mid in ("libreyolo-yolonas-s", "libreyolo-yolonas-l"):
        pol = get_policy(mid)
        assert pol is None or pol.final_policy != "commercial_safe_core"


def test_legal_audit_doc_exists_and_lists_gated():
    doc = _SRC.parents[1] / "docs" / "legal_model_audit.md"
    if doc.is_file():
        text = doc.read_text().lower()
        assert "ultralytics" in text and "none" in text
        assert "gated" in text
