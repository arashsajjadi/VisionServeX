# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7: interactive segmentation runtime (classic runs) or honest blockers (deep)."""

from __future__ import annotations

import pytest


def test_vsx_interactive_returns_handle():
    from visionservex.vsx import VSX

    h = VSX.interactive("ritm")
    assert h.family == "interactive"
    assert hasattr(h, "explain") and callable(h)


def test_ritm_state_checkpoint_required():
    from visionservex.vsx import VSX

    assert VSX.interactive("ritm").status() == "checkpoint_required"


@pytest.mark.parametrize("m", ["clickseg", "simpleclick", "focalclick"])
def test_legal_review_models(m):
    from visionservex.vsx import VSX

    info = VSX.interactive(m).explain()
    assert info["state"] == "legal_review_required"
    assert info["commercial_safe"] is False


def test_deep_model_without_checkpoint_raises_structured():
    from PIL import Image

    from visionservex.vsx import VSX, VSXError

    img = Image.new("RGB", (64, 64))
    with pytest.raises(VSXError) as exc:
        VSX.interactive("ritm")(img, positive_points=[(30, 30)])
    assert exc.value.state in ("checkpoint_required",)
    assert "git clone" in exc.value.next_command or "ritm" in exc.value.next_command.lower()


def test_classic_grabcut_actually_runs():
    pytest.importorskip("cv2", reason="opencv-python not installed")
    import numpy as np
    from PIL import Image

    from visionservex.vsx import VSX

    img = Image.fromarray((np.random.rand(80, 80, 3) * 255).astype("uint8"))
    res = VSX.interactive("grabcut")(img, positive_points=[(40, 40)], negative_points=[(5, 5)])
    assert res["status"] == "ok"
    assert res["backend"] == "opencv-grabcut"
    assert res["mask_area"] >= 0
    assert res["mask"] is not None


def test_simpleclick_commercial_unsafe_documented():
    from visionservex.interactive_runtime import facts

    f = facts("simpleclick")
    assert f["commercial_safe"] is False
    assert "MAE" in f["training_data"] or "CC-BY-NC" in f["training_data"]


def test_interactive_list_has_eight():
    from visionservex.interactive_runtime import _CLASSIC, _FACTS

    assert len(_FACTS) == 4
    assert len(_CLASSIC) >= 4
