"""Tests for the classic (weight-free) smart-annotation refiners.

Fast, CPU-only, deterministic — runs in the quick-safe suite.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("skimage")
pytest.importorskip("sklearn")

from visionservex.smart_annotation import (  # noqa: E402
    METHOD_LICENSE,
    Prompt,
    list_methods,
    refine,
)
from visionservex.smart_annotation.contracts import PROMPT_MODALITIES  # noqa: E402
from visionservex.smart_annotation.metrics import (  # noqa: E402
    boundary_iou,
    iou,
    success_rate_at_iou,
)
from visionservex.smart_annotation.prompts import build_prompt, parse_box  # noqa: E402

EXPECTED_METHODS = {
    "classic-grabcut",
    "classic-marker-watershed",
    "classic-random-walker",
    "classic-slic-graphcut",
    "classic-intelligent-scissors",
    "classic-interactive-rf",
    "classic-slic-rf-smooth",
    "classic-edge-plus",
}


def _synthetic():
    rng = np.random.default_rng(0)
    h = w = 128
    img = rng.normal(120, 10, (h, w, 3)).clip(0, 255).astype("uint8")
    gt = np.zeros((h, w), "uint8")
    cv2.circle(gt, (64, 64), 34, 1, -1)
    img[gt > 0] = (40, 200, 60)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    poly = [
        [64 + 34 * np.cos(a), 64 + 34 * np.sin(a)]
        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ]
    prompt = Prompt(
        box=(64 - 38, 64 - 38, 64 + 38, 64 + 38),
        positive_points=[[64, 64], [64, 48], [80, 64]],
        negative_points=[[6, 6], [120, 120], [6, 120], [120, 6]],
        polygon=poly,
        polyline=poly,
    )
    return img, gt, prompt


def test_all_eight_methods_registered():
    assert set(list_methods()) == EXPECTED_METHODS
    assert len(METHOD_LICENSE) == 8


@pytest.mark.parametrize("method", sorted(EXPECTED_METHODS))
def test_method_runs_and_returns_contract(method):
    img, gt, prompt = _synthetic()
    r = refine(img, prompt, method=method)
    # contract
    d = r.to_contract_dict()
    for k in ("polygon", "bbox", "method", "latency_ms", "device", "license_safe"):
        assert k in d, k
    assert r.device == "cpu"
    assert r.license_safe is True
    assert r.method == method
    # non-empty, sane mask
    assert r.mask.dtype == np.uint8
    assert r.mask.shape == gt.shape
    assert r.mask.sum() > 0, f"{method} produced an empty mask"
    # on a trivially-separable synthetic object every classic method should clear 0.5 IoU
    assert iou(r.mask, gt) >= 0.5, f"{method} IoU too low on synthetic object"


def test_license_safe_dependencies_only():
    # No method may advertise a copyleft (GPL/AGPL) dependency.
    for name, lic in METHOD_LICENSE.items():
        assert "GPL" not in lic.upper(), f"{name} pulls a copyleft dep: {lic}"
        assert any(tag in lic for tag in ("Apache-2.0", "BSD-3")), f"{name}: {lic}"


def test_metrics_basics():
    a = np.zeros((10, 10), "uint8")
    a[2:8, 2:8] = 1
    assert iou(a, a) == 1.0
    assert iou(a, np.zeros_like(a)) == 0.0
    assert boundary_iou(a, a) == 1.0
    assert success_rate_at_iou([0.9, 0.4, 0.8], 0.5) == pytest.approx(2 / 3)


def test_prompt_parsing():
    assert parse_box("10,20,200,220") == (10.0, 20.0, 200.0, 220.0)
    assert parse_box(None) is None
    p = build_prompt(box="0,0,10,10", positive_points=[[5, 5]])
    assert p.box == (0.0, 0.0, 10.0, 10.0)
    assert p.positive_points == [[5.0, 5.0]]
    assert not p.is_empty()
    assert Prompt().is_empty()
    assert set(PROMPT_MODALITIES) >= {"box", "polygon", "positive_points", "scribble", "mask_hint"}


def test_unknown_method_and_empty_prompt_raise():
    img, _gt, prompt = _synthetic()
    with pytest.raises(KeyError):
        refine(img, prompt, method="classic-does-not-exist")
    with pytest.raises(ValueError):
        refine(img, Prompt(), method="classic-grabcut")
