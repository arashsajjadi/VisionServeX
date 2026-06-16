# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.15.0: registry/policy sync + legal firewall invariants.

Enforces the v3.15 legal contract: every default-safe policy row is permissive
(no AGPL/GPL), the new torchvision family has matching registry + policy rows,
YOLO-NAS stays blocked, and no runtime engine imports Ultralytics/AGPL.
"""

from __future__ import annotations

from pathlib import Path

from visionservex.licensing.policy import _ROWS, get_policy
from visionservex.registry import default_registry

_TV_IDS = [
    "torchvision-alexnet",
    "torchvision-resnet18",
    "torchvision-resnet34",
    "torchvision-resnet50",
    "torchvision-resnet101",
    "torchvision-resnet152",
    "torchvision-wide-resnet50-2",
    "torchvision-resnext50-32x4d",
    "torchvision-densenet121",
    "torchvision-mobilenet-v2",
    "torchvision-mobilenet-v3-large",
    "torchvision-efficientnet-b0",
    "torchvision-convnext-tiny",
]

# Engines that constitute the runtime; none may import Ultralytics/AGPL.
_ENGINES_DIR = Path(__file__).resolve().parents[1] / "src" / "visionservex" / "engines"
_RUNTIME_DIRS = [
    _ENGINES_DIR,
    Path(__file__).resolve().parents[1] / "src" / "visionservex" / "core",
    Path(__file__).resolve().parents[1] / "src" / "visionservex" / "data",
]


def test_torchvision_rows_have_registry_and_policy():
    reg = default_registry()
    for mid in _TV_IDS:
        entry = reg.get(mid)  # raises if missing
        assert entry.license == "BSD-3-Clause"
        pol = get_policy(mid)
        assert pol is not None, f"{mid} has registry row but no policy row"
        assert pol.final_policy == "commercial_safe_core"
        assert pol.default_safe is True
        assert pol.can_ship_weights is False  # never bundle weights
        assert "GPL" not in (pol.code_license or "")
        assert "GPL" not in (pol.weights_license or "")


def test_no_default_safe_policy_is_agpl_or_gpl():
    """No commercial_safe_core / default-safe policy row may carry an AGPL/GPL license."""
    for r in _ROWS:
        if r.final_policy == "commercial_safe_core" or r.default_safe:
            for lic in (r.code_license or "", r.weights_license or ""):
                up = lic.upper()
                assert "AGPL" not in up and "GPL-3" not in up and "GPLV3" not in up, (
                    f"{r.model_id} is default-safe but license={lic!r}"
                )


def test_yolonas_never_default_safe_or_trainable():
    from visionservex.core.model import _training_capabilities
    from visionservex.engines.libreyolo import _TRAINABLE_FAMILIES

    assert "yolonas" not in _TRAINABLE_FAMILIES
    for mid in ("libreyolo-yolonas-s", "libreyolo-yolonas-l"):
        cap = _training_capabilities(mid)
        assert cap["train_supported"] is False
        # not in commercial_safe_core
        pol = get_policy(mid)
        assert pol is None or pol.final_policy != "commercial_safe_core"


def test_no_ultralytics_or_agpl_in_runtime_engines():
    forbidden = ("import ultralytics", "from ultralytics", "ultralytics.YOLO")
    for d in _RUNTIME_DIRS:
        for f in d.rglob("*.py"):
            text = f.read_text()
            for pat in forbidden:
                assert pat not in text, f"{f} imports Ultralytics ({pat!r})"


def test_commercial_safe_core_count_includes_torchvision():
    core = [r for r in _ROWS if r.final_policy == "commercial_safe_core"]
    tv = [r for r in core if r.family == "torchvision-classify"]
    assert len(tv) == 13, f"expected 13 torchvision commercial_safe_core rows, got {len(tv)}"
