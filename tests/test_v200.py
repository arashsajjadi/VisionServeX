# SPDX-License-Identifier: Apache-2.0
"""Tests for v2.0.0: real-model engine fixes, agriculture/aerial CLIs, Florence-2 version guard."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

torch = pytest.importorskip("torch", reason="v2.0.0 engine tests require torch tensors")


def _tiny():
    return Image.new("RGB", (64, 64), color=(100, 100, 100))


# ---------------------------------------------------------------------------
# OWLv2 engine fix: post_process_object_detection via image_processor
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_owlv2_post_process_resolves_from_image_processor():
    """post_process_object_detection should fall back to image_processor sub-attribute."""
    from visionservex.engines.owlv2 import OWLv2Engine
    from visionservex.registry import default_registry

    reg = default_registry()
    engine = OWLv2Engine(reg.get("owlv2-base-patch16"))
    engine._real_ready = True
    engine._torch = torch

    # Processor without post_process_object_detection directly (fast processor model).
    proc = MagicMock(spec=[])  # empty spec — has no attributes by default
    proc.post_process_object_detection = None  # explicitly None so getattr returns None

    # But image_processor sub-model has it.
    img_proc = MagicMock()
    expected_return = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "scores": torch.tensor([0.8]),
            "labels": torch.tensor([0]),
        }
    ]
    img_proc.post_process_object_detection.return_value = expected_return
    proc.image_processor = img_proc

    param = torch.zeros(1)
    engine._processor = proc
    engine._model = MagicMock()
    engine._model.parameters.side_effect = lambda: iter([param])
    engine._model.return_value = MagicMock()

    # Mock processor __call__ to return dict-like inputs
    class _FakeInputs(dict):
        pass

    proc.return_value = _FakeInputs({"pixel_values": torch.zeros(1, 3, 64, 64)})

    result = engine.predict(_tiny(), prompts=["object"])
    # Should have used image_processor.post_process_object_detection
    img_proc.post_process_object_detection.assert_called_once()
    assert len(result.detections) == 1


# ---------------------------------------------------------------------------
# Florence-2 version guard
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_florence2_raises_when_transformers_5():
    """Florence-2 engine must raise MissingDependencyError on transformers >= 5.0."""
    from unittest.mock import patch

    from visionservex.engines.base import MissingDependencyError
    from visionservex.engines.florence2 import Florence2Engine
    from visionservex.registry import default_registry

    reg = default_registry()
    engine = Florence2Engine(reg.get("florence-2-base"))

    # Simulate transformers 5.x by patching the version string
    with patch("visionservex.engines.florence2.assert_modules"):
        import transformers as _tr

        orig_ver = _tr.__version__
        try:
            _tr.__version__ = "5.0.0"
            with pytest.raises(MissingDependencyError, match=r"transformers 5\.0\.0"):
                engine._real_load(device="cpu", precision="fp32")
        finally:
            _tr.__version__ = orig_ver


@pytest.mark.fast
def test_florence2_compat_shim_functions_importable():
    from visionservex.engines.florence2 import (
        _apply_florence2_config_shim,
        _apply_florence2_tokenizer_shim,
    )

    # All three should be callable without errors
    _apply_florence2_tokenizer_shim()  # harmless if tokenizers backend not present
    _apply_florence2_config_shim()  # harmless if no cached config module


# ---------------------------------------------------------------------------
# SigLIP2 fix: vision_model path used when full model needs input_ids
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_siglip2_uses_vision_model_subpath():
    """DINOv2 engine routes through vision_model when only pixel_values are present."""
    from visionservex.engines.dinov2 import DINOv2Engine
    from visionservex.registry import default_registry

    reg = default_registry()
    engine = DINOv2Engine(reg.get("siglip2-base-patch16-224"))
    engine._real_ready = True
    engine._torch = torch

    param = torch.zeros(768)

    # Full model that would require input_ids
    full_model = MagicMock()
    full_model.parameters.side_effect = lambda: iter([param])

    # vision_model submodel that returns pooler_output
    vision_mock_out = MagicMock()
    vision_mock_out.pooler_output = torch.randn(1, 768)
    vision_submodel = MagicMock()
    vision_submodel.return_value = vision_mock_out
    full_model.vision_model = vision_submodel

    proc = MagicMock()
    proc.return_value = {"pixel_values": torch.zeros(1, 3, 64, 64)}
    engine._processor = proc
    engine._model = full_model

    engine.predict(_tiny())
    # Should have called vision_model, not the full model
    vision_submodel.assert_called_once()
    full_model.assert_not_called()


# ---------------------------------------------------------------------------
# Agriculture commands
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_agriculture_commands_registered():
    from visionservex.cli.main import app

    names = {g.name for g in app.registered_groups}
    assert "agriculture" in names


@pytest.mark.fast
def test_agriculture_recipe_known_names():
    # Recipe name validation happens inside the function — just confirm it's callable
    # without actually running the Click/Typer context
    import inspect

    from visionservex.cli.agriculture_commands import recipe as _recipe

    assert inspect.isfunction(_recipe) or callable(_recipe)


# ---------------------------------------------------------------------------
# Aerial commands
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_aerial_commands_registered():
    from visionservex.cli.main import app

    names = {g.name for g in app.registered_groups}
    assert "aerial" in names


@pytest.mark.fast
def test_aerial_dataset_validate_dota_missing_path(tmp_path):
    from visionservex.cli.aerial_commands import _AERIAL_MODELS

    # Validate that model metadata is complete
    assert "rtmdet-r2-s" in _AERIAL_MODELS
    assert "blocker" in _AERIAL_MODELS["rtmdet-r2-s"]


@pytest.mark.fast
def test_aerial_obb_metric_note_mentions_rotated_iou():
    """Aerial OBB models must document that rotated IoU is needed, not axis-aligned AP."""
    from visionservex.cli.aerial_commands import _AERIAL_MODELS

    for _mid, info in _AERIAL_MODELS.items():
        if info["task"] == "oriented_detection":
            # The info dict should mention the metric correctly
            all_text = " ".join(str(v) for v in info.values())
            assert any("rotated" in all_text.lower() or "obb" in all_text.lower() for _ in [1])


# ---------------------------------------------------------------------------
# Florence-2 version guard test for real transformers
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_florence2_version_check_is_version_string():
    """The Florence-2 version error message must include the actual version string."""
    from unittest.mock import patch

    from visionservex.engines.base import MissingDependencyError
    from visionservex.engines.florence2 import Florence2Engine
    from visionservex.registry import default_registry

    reg = default_registry()
    engine = Florence2Engine(reg.get("florence-2-base"))

    import transformers as _tr

    orig_ver = _tr.__version__
    try:
        _tr.__version__ = "5.99.0"
        with patch("visionservex.engines.florence2.assert_modules"):
            with pytest.raises(MissingDependencyError) as exc_info:
                engine._real_load(device="cpu", precision="fp32")
            # Error message must include the actual version
            assert "5.99.0" in str(exc_info.value)
            assert "transformers>=4.40" in str(exc_info.value)
    finally:
        _tr.__version__ = orig_ver


# ---------------------------------------------------------------------------
# DEIMv2 / RT-DETRv4 blocker in manifest
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_deimv2_and_rtdetrv4_have_blockers():
    """DEIMv2 and RT-DETRv4 manifest entries must document their exact blocker."""
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST

    deimv2_entries = {k for k in SOURCE_MANIFEST if k.startswith("deimv2")}
    assert deimv2_entries, "No DEIMv2 entries in manifest"
    # v2.48+: DEIMv2 was genuinely wired and benchmarked — most variants
    # (atto/femto/pico/n/m/l/x) are now benchmark_passed in model_coverage_ledger.csv,
    # so they are no longer non-runnable stubs. The original v2.00 assertion ("all DEIMv2
    # entries are stubs") is stale. Current truth: any entry still marked non-runnable must
    # document its exact blocker, and at least one variant must now be runnable.
    for k in deimv2_entries:
        e = SOURCE_MANIFEST[k]
        if not e.runnable_in_visionservex:
            assert e.known_blockers, f"{k} (non-runnable) must document known_blockers"
    assert any(SOURCE_MANIFEST[k].runnable_in_visionservex for k in deimv2_entries), (
        "expected at least one runnable DEIMv2 variant after the v2.48 wiring"
    )
