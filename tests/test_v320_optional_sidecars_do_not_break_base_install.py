# SPDX-License-Identifier: Apache-2.0
"""v3.20: optional sidecars/extras never break the base capability path. Weight-free."""

from __future__ import annotations

import sys
from pathlib import Path

from visionservex.core.model import list_models, model_capabilities

_SRC = Path(__file__).resolve().parents[1] / "src" / "visionservex"
_HEAVY = (
    "torch",
    "transformers",
    "rfdetr",
    "timm",
    "faster_coco_eval",
    "pytorch_lightning",
    "mmcv",
    "mmdet",
    "mmpose",
)


def test_capabilities_work_for_all_models_without_heavy_imports():
    ids = list_models()
    assert len(ids) == 151
    for m in ids:
        assert model_capabilities(m)["readiness_state"]


def test_readiness_and_training_modules_have_no_top_level_heavy_imports():
    for rel in ("readiness/taxonomy.py", "readiness/live_evidence.py"):
        text = (_SRC / rel).read_text()
        for dep in _HEAVY:
            assert f"\nimport {dep}" not in text and f"\nfrom {dep}" not in text, (rel, dep)


def test_embedding_finetune_imports_torch_lazily():
    # torch must be imported inside functions, not at module top, so the base
    # capability import stays light.
    text = (_SRC / "training" / "embedding_finetune.py").read_text()
    head = text.split("def ")[0]
    assert "\nimport torch" not in head and "\nfrom torch" not in head


def test_no_sidecar_dependency_leaks_into_base_import():
    for dep in ("faster_coco_eval", "mmcv", "mmdet", "mmpose", "pytorch_lightning"):
        assert dep not in sys.modules, f"{dep} leaked into the base import path"
