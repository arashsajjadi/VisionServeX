# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.30.0: smoke-matrix LibreYOLO default-safe integration."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_LIBREYOLO_AVAILABLE = importlib.util.find_spec("libreyolo") is not None
pytestmark = pytest.mark.skipif(not _LIBREYOLO_AVAILABLE, reason="libreyolo not installed")


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


def test_libreyolo_default_safe_includes_yolo9() -> None:
    """yolo9 is permissive MIT and IS allowed in default-safe.

    v2.48 relicensing audit (re-affirmed by the v3.3 truth audit): LibreYOLO pulls yolo9
    weights from the MIT-licensed MultimediaTechLab/YOLO upstream, NOT the original
    WongKinYiu/yolov9 GPL-3.0 repo — see ``test_yolo9_is_permissive_mit``. This was
    previously an exclusion guard from the pre-v2.48 GPL era; it never got updated when the
    relicensing was decided, so it had been failing. Updated to the project's current truth:
    yolo9 (MIT) may be default-safe; the genuinely non-commercial family (yolonas) is the one
    that must stay excluded (guarded by ``test_libreyolo_default_safe_excludes_yolonas``).
    """
    from run_model_smoke_matrix import _get_libreyolo_default_safe_models

    models = _get_libreyolo_default_safe_models()
    families = {m.get("family") for m in models}
    # The non-commercial family must never be default-safe.
    assert "yolonas" not in families, f"YOLO-NAS leaked into default-safe: {families}"
    # yolo9 (MIT via MultimediaTechLab/YOLO) is permissive and present in default-safe.
    assert "yolo9" in families, "yolo9 (MIT, v2.48 relicensing) should be in default-safe"
