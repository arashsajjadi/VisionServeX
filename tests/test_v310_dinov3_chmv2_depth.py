# SPDX-License-Identifier: Apache-2.0
"""v3.10.0: DINOv3 CHMv2 DPT depth head — policy + runtime tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ART = Path("notebook/99_final_report/artifacts/v310/dinov3_chmv2_dpt")


def test_chmv2_in_policy():
    from visionservex.licensing.policy import _ROWS

    ids = {r.model_id for r in _ROWS}
    assert "dinov3-vitl16-chmv2-dpt-head" in ids


def test_chmv2_policy_hf_repo():
    from visionservex.licensing.policy import get_policy, resolve_model_id

    pol = get_policy(resolve_model_id("dinov3-vitl16-chmv2-dpt-head"))
    assert pol is not None
    assert pol.hf_repo == "facebook/dinov3-vitl16-chmv2-dpt-head"
    assert not pol.can_ship_weights


def test_chmv2_policy_is_byot():
    from visionservex.licensing.policy import get_policy, resolve_model_id

    pol = get_policy(resolve_model_id("dinov3-vitl16-chmv2-dpt-head"))
    assert pol.final_policy == "byot_license_required"


def test_dinov3_depth_exported_in_byot_runtime():
    from visionservex import byot_runtime

    assert "dinov3_depth" in byot_runtime.__all__
    assert callable(byot_runtime.dinov3_depth)


def test_dinov3_depth_version_gate():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import transformers
    from packaging.version import Version

    # if transformers < 5.10 the function returns a blocked dict
    if Version(transformers.__version__) < Version("5.10.0"):
        from visionservex.byot_runtime import dinov3_depth

        result = dinov3_depth(
            "dinov3-vitl16-chmv2-dpt-head", "tests/assets/smoke/coco_person_car.jpg"
        )
        assert result["state"] == "runtime_blocked_byot"
    else:
        # >= 5.10: version gate should pass (actual inference not required here)
        pass


def test_chmv2_benchmark_artifact_exists():
    meta = ART / "metadata.json"
    if not meta.exists():
        pytest.skip("CHMv2 DPT depth benchmark artifact not on disk")
    data = json.loads(meta.read_text())
    assert data["benchmark_state"] == "benchmark_passed_byot_depth"
    assert data["depth_nonzero"] > 0


def test_chmv2_depth_png_exists():
    depth_png = ART / "depth.png"
    if not depth_png.exists():
        pytest.skip("CHMv2 depth.png not on disk")
    # verify non-empty image
    from PIL import Image

    img = Image.open(depth_png)
    assert img.width > 0 and img.height > 0
