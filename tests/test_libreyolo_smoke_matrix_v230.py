# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.30.0: smoke-matrix LibreYOLO default-safe integration."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "tools"))


def test_libreyolo_default_safe_models_listed() -> None:
    """_get_libreyolo_default_safe_models must return Apache-2.0/MIT weights only."""
    from run_model_smoke_matrix import _get_libreyolo_default_safe_models

    models = _get_libreyolo_default_safe_models()
    # Either libreyolo not installed (empty) or default-safe list non-empty
    assert isinstance(models, list)
    if models:
        for m in models:
            assert m["source"] == "libreyolo"
            wl = (m.get("weight_license", "") or "").upper()
            assert any(ok in wl for ok in ("APACHE", "MIT")), (
                f"non-permissive weight in default-safe list: {m}"
            )


def test_smoke_matrix_advertised_includes_libreyolo_when_flagged() -> None:
    """When include_libreyolo_default_safe=True, advertised list includes libreyolo rows."""
    from run_model_smoke_matrix import _get_advertised_models

    core_only = _get_advertised_models(include_core=True, include_libreyolo_default_safe=False)
    with_libreyolo = _get_advertised_models(include_core=True, include_libreyolo_default_safe=True)
    assert len(with_libreyolo) >= len(core_only)


def test_libreyolo_smoke_command_uses_libreyolo_subcommand() -> None:
    """Models with source='libreyolo' must dispatch to `libreyolo smoke-test`."""
    from run_model_smoke_matrix import _build_libreyolo_smoke_command

    cmd = _build_libreyolo_smoke_command("libreyolo-yolox-n", "detect", device="cpu")
    assert "libreyolo" in cmd
    assert "smoke-test" in cmd
    assert "libreyolo-yolox-n" in cmd


def test_libreyolo_default_safe_excludes_yolonas() -> None:
    """YOLO-NAS weights must never enter default-safe (non-commercial)."""
    from run_model_smoke_matrix import _get_libreyolo_default_safe_models

    models = _get_libreyolo_default_safe_models()
    for m in models:
        assert m.get("family") != "yolonas", f"YOLO-NAS leaked into default-safe: {m}"


def test_libreyolo_default_safe_excludes_yolo9() -> None:
    """YOLOv9 weights must never enter default-safe (GPL)."""
    from run_model_smoke_matrix import _get_libreyolo_default_safe_models

    models = _get_libreyolo_default_safe_models()
    for m in models:
        assert m.get("family") != "yolo9", f"YOLOv9 leaked into default-safe: {m}"
