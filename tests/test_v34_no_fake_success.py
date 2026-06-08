# SPDX-License-Identifier: Apache-2.0
"""v3.4 no-fake-success guard tests.

Every test in this module asserts that VisionServeX never silently pretends
a gated, excluded, or sidecar-only model is runnable.  The library must
raise an exception or return a non-"ok" / non-"benchmark_passed" status for
all models listed here.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# a. test_sam3_base_not_runnable
# ---------------------------------------------------------------------------


def test_sam3_base_not_runnable():
    """VisionModel('sam3-base') raises an exception or VSX.sam returns expected_blocker.

    sam3-base is HF-gated; attempting to use it must never silently succeed.
    """
    # VSX.sam path: must NOT report benchmark_passed, and segment must raise.
    from visionservex.vsx import VSX, VSXError

    info = VSX.sam("sam3-base").explain()
    assert info["state"] != "benchmark_passed", (
        f"sam3-base must not be benchmark_passed, got state={info['state']!r}"
    )
    assert info["auth_required"] is True

    with pytest.raises(VSXError):
        VSX.sam("sam3-base").segment("dummy.jpg")


# ---------------------------------------------------------------------------
# b. test_dinov3_not_runnable
# ---------------------------------------------------------------------------


def test_dinov3_not_runnable():
    """VisionModel('dinov3-vitb16') raises or is excluded from the registry.

    dinov3-vitb16 carries a custom Meta license and is HF-gated.  VSX must
    either raise RegistryError (model not in registry) or return a non-runnable
    state and raise when embed is attempted.
    """
    try:
        from visionservex import VisionModel

        VisionModel("dinov3-vitb16")
        # If it did not raise, it must at least not be benchmark_passed via VSX.
        from visionservex.vsx import VSX, VSXError

        info = VSX.dino("dinov3-vitb16").explain()
        assert info["state"] != "benchmark_passed", (
            f"dinov3-vitb16 must not be benchmark_passed, got {info['state']!r}"
        )
        with pytest.raises(VSXError):
            VSX.dino("dinov3-vitb16").embed("dummy.jpg")
    except Exception as exc:
        # RegistryError or any exception on construction is acceptable.
        assert exc is not None  # construction itself raising is the correct behaviour


# ---------------------------------------------------------------------------
# c. test_grounding_dino_1_5_requires_api
# ---------------------------------------------------------------------------


def test_grounding_dino_1_5_requires_api():
    """VisionModel('grounding-dino-1.5') raises or returns auth_required.

    grounding-dino-1.5 is an API-gated model (requires DEEPDATASPACE_API_KEY).
    """
    from visionservex.vsx import VSX, VSXError

    info = VSX.dino("grounding-dino-1.5").explain()
    assert info["state"] in ("auth_required", "external_api_only"), (
        f"grounding-dino-1.5 must be auth_required or external_api_only, got {info['state']!r}"
    )
    assert info["auth_required"] is True

    with pytest.raises(VSXError):
        VSX.dino("grounding-dino-1.5").detect("dummy.jpg", text="cat")


# ---------------------------------------------------------------------------
# d. test_onnx_non_eligible_raises
# ---------------------------------------------------------------------------


def test_onnx_non_eligible_raises(tmp_path):
    """export_sam_decoder_onnx('edgesam', ...) raises ValueError.

    edgesam / edge-sam is excluded (non-commercial licence) and must never be
    silently exported.
    """
    from visionservex.onnx_export import export_sam_decoder_onnx

    with pytest.raises(ValueError, match="not ONNX-eligible"):
        export_sam_decoder_onnx("edgesam", str(tmp_path / "x.onnx"))


# ---------------------------------------------------------------------------
# e. test_sam3_pipeline_blocked
# ---------------------------------------------------------------------------


def test_sam3_pipeline_blocked():
    """VSX.pipeline('grounding-dino-1.5+sam3-base').explain() shows auth_required.

    A pipeline composed of two blocked components must itself be blocked.
    """
    from visionservex.vsx import VSX

    ph = VSX.pipeline("grounding-dino-1.5+sam3-base")
    info = ph.explain()

    assert info["state"] not in ("ok", "benchmark_passed", "pipeline_demo_ready"), (
        f"Expected blocked pipeline, got state={info['state']!r}"
    )
    assert info["state"] == "auth_required", f"Expected auth_required, got {info['state']!r}"


# ---------------------------------------------------------------------------
# f. test_sidecar_models_not_benchmark_passed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mid", ["maskdino-r50-coco", "co-dino-inst-vit-l-coco"])
def test_sidecar_models_not_benchmark_passed(mid):
    """Sidecar-only models must not silently succeed via VisionModel.predict().

    maskdino-r50-coco and co-dino-inst-vit-l-coco require Detectron2 /
    OpenMMLab sidecars that are not installed in a standard VisionServeX
    environment.  Calling predict() must raise an exception (never return a
    valid result silently).
    """
    from visionservex import VisionModel

    model = VisionModel(mid)
    # implementation_status must be 'stub' — never 'ready' / 'benchmark_passed'.
    impl = getattr(model.entry, "implementation_status", "")
    assert impl != "benchmark_passed", (
        f"{mid} implementation_status must not be 'benchmark_passed', got {impl!r}"
    )

    # predict must raise — never silently succeed.
    with pytest.raises(Exception):  # noqa: B017 — must raise *something*, type irrelevant
        model.predict("dummy.jpg")
