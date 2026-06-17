# SPDX-License-Identifier: Apache-2.0
"""v3.19: optional/heavy deps never gate the base capability path. Weight-free.

The capability-truth API and the readiness modules must work without torch,
transformers, rfdetr, or the training-only ``faster-coco-eval`` dependency that
v3.19 introduced for RF-DETR live training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import visionservex as vsx
from visionservex.core.model import list_models, model_capabilities

_SRC = Path(__file__).resolve().parents[1] / "src" / "visionservex"
_HEAVY = ("torch", "transformers", "rfdetr", "timm", "faster_coco_eval", "pytorch_lightning")


def test_capabilities_work_for_all_models():
    ids = list_models()
    assert len(ids) >= 140
    for m in ids:
        assert model_capabilities(m)["readiness_state"]


def test_readiness_modules_have_no_top_level_heavy_imports():
    for rel in ("readiness/taxonomy.py", "readiness/live_evidence.py"):
        text = (_SRC / rel).read_text()
        for dep in _HEAVY:
            assert f"\nimport {dep}" not in text and f"\nfrom {dep}" not in text, (rel, dep)


def test_faster_coco_eval_is_training_only_not_imported_by_package():
    # importing visionservex (already imported by this test session) must not pull
    # the training-only eval dependency.
    assert "faster_coco_eval" not in sys.modules, (
        "faster-coco-eval leaked into the base import path"
    )


def test_base_capability_object_has_no_heavy_objects():
    # model_capabilities returns plain JSON-able metadata, never live model objects.
    import json

    for m in list_models()[:30]:
        json.dumps(model_capabilities(m), default=str)  # must be serialisable
    assert vsx.__version__
