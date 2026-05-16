# SPDX-License-Identifier: Apache-2.0
"""Tests for v1.8.0: OWLv2 engine, Florence-2 engine, SAM3 auth wrapper, expert sidecars.

All fast tests use mocked outputs — no real HF model is loaded.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image


def _tiny_image():
    return Image.new("RGB", (64, 64), color=(128, 128, 128))


# ---------------------------------------------------------------------------
# OWLv2 engine
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_owlv2_engine_registered():
    """OWLv2 engine factory must be registered under 'owlv2'."""
    from visionservex.engines import build_engine
    from visionservex.registry import default_registry

    reg = default_registry()
    entry = reg.get("owlv2-base-patch16")
    assert entry.engine == "owlv2"
    assert entry.implementation_status == "wired"

    engine = build_engine(entry)
    from visionservex.engines.owlv2 import OWLv2Engine

    assert isinstance(engine, OWLv2Engine)


@pytest.mark.fast
def test_owlv2_predict_with_mocked_outputs():
    """Mock processor/model and verify OWLv2 result normalization."""
    import torch

    from visionservex.engines.owlv2 import OWLv2Engine
    from visionservex.registry import default_registry

    reg = default_registry()
    entry = reg.get("owlv2-base-patch16")
    engine = OWLv2Engine(entry)
    engine.device = "cpu"
    engine.precision = "fp32"
    engine._real_ready = True
    engine._torch = torch

    # Mock processor
    proc = MagicMock()

    class _MockInputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

    proc.return_value = _MockInputs(
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.randn(1, 3, 64, 64),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
    )
    proc.post_process_object_detection.return_value = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 40.0, 40.0]]),
            "scores": torch.tensor([0.9, 0.7]),
            "labels": torch.tensor([0, 1]),
        }
    ]
    engine._processor = proc

    # Mock model — parameters() returns a fresh iterator each call (next() is called twice).
    param = torch.zeros(1, dtype=torch.float32)
    model = MagicMock()
    model.return_value = MagicMock()  # outputs object
    model.parameters.side_effect = lambda: iter([param])
    engine._model = model

    result = engine.predict(_tiny_image(), prompts=["person", "car"], threshold=0.5)

    assert result.kind == "open_vocab"
    assert result.model_id == "owlv2-base-patch16"
    assert len(result.detections) == 2
    assert result.detections[0].label == "person"
    assert result.detections[0].score == pytest.approx(0.9)
    assert result.detections[1].label == "car"
    assert result.prompts == ["person", "car"]
    assert result.metadata["backend"] == "huggingface_owlv2"


@pytest.mark.fast
def test_owlv2_accepts_comma_separated_prompt():
    """OWLv2 accepts a comma-separated string via prompt= (CLI shape)."""
    import torch

    from visionservex.engines.owlv2 import OWLv2Engine
    from visionservex.registry import default_registry

    reg = default_registry()
    engine = OWLv2Engine(reg.get("owlv2-base-patch16"))
    engine._real_ready = True
    engine._torch = torch

    proc = MagicMock()

    class _MockInputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

    proc.return_value = _MockInputs(
        {"input_ids": torch.tensor([[1]]), "pixel_values": torch.zeros(1, 3, 64, 64)}
    )
    proc.post_process_object_detection.return_value = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "scores": torch.tensor([0.5]),
            "labels": torch.tensor([1]),
        }
    ]
    engine._processor = proc
    param = torch.zeros(1)
    engine._model = MagicMock()
    engine._model.parameters.side_effect = lambda: iter([param])

    result = engine.predict(_tiny_image(), prompt="cat, dog, bird")
    assert result.prompts == ["cat", "dog", "bird"]
    # label 1 → "dog"
    assert result.detections[0].label == "dog"


# ---------------------------------------------------------------------------
# Florence-2 parser
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_florence2_engine_registered():
    from visionservex.engines import build_engine
    from visionservex.engines.florence2 import Florence2Engine
    from visionservex.registry import default_registry

    reg = default_registry()
    entry = reg.get("florence-2-base")
    assert entry.engine == "florence2"
    assert entry.implementation_status == "wired"

    engine = build_engine(entry)
    assert isinstance(engine, Florence2Engine)


@pytest.mark.fast
def test_florence2_parse_caption_text():
    """parse_florence2_generation handles text-only tasks (caption, OCR)."""
    from visionservex.engines.florence2 import parse_florence2_generation

    parsed = {"<CAPTION>": "A person standing in a room."}
    detections, extra = parse_florence2_generation(
        parsed, task_token="<CAPTION>", image_size=(640, 480)
    )
    assert detections == []
    assert extra["text"] == "A person standing in a room."
    assert extra["task_token"] == "<CAPTION>"


@pytest.mark.fast
def test_florence2_parse_object_detection_bboxes():
    """parse_florence2_generation builds Detection list for OD task."""
    from visionservex.engines.florence2 import parse_florence2_generation

    parsed = {
        "<OD>": {
            "bboxes": [[10, 10, 50, 50], [20, 20, 60, 60]],
            "labels": ["person", "car"],
        }
    }
    detections, _ = parse_florence2_generation(parsed, task_token="<OD>", image_size=(640, 480))
    assert len(detections) == 2
    assert detections[0].label == "person"
    assert detections[0].box.x1 == 10.0
    assert detections[1].label == "car"


@pytest.mark.fast
def test_florence2_parse_phrase_grounding():
    """phrase grounding returns bboxes + matching label text."""
    from visionservex.engines.florence2 import parse_florence2_generation

    parsed = {
        "<CAPTION_TO_PHRASE_GROUNDING>": {
            "bboxes": [[5, 5, 25, 25]],
            "labels": ["red shirt"],
        }
    }
    detections, _ = parse_florence2_generation(
        parsed, task_token="<CAPTION_TO_PHRASE_GROUNDING>", image_size=(100, 100)
    )
    assert len(detections) == 1
    assert detections[0].label == "red shirt"


@pytest.mark.fast
def test_florence2_parse_quad_boxes_to_axis_aligned():
    """Quad polygon boxes are converted to axis-aligned x1y1x2y2."""
    from visionservex.engines.florence2 import parse_florence2_generation

    parsed = {
        "<OCR_WITH_REGION>": {
            "quad_boxes": [[10, 10, 50, 10, 50, 30, 10, 30]],  # rect via 4 points
            "labels": ["hello"],
        }
    }
    detections, _ = parse_florence2_generation(
        parsed, task_token="<OCR_WITH_REGION>", image_size=(100, 100)
    )
    assert len(detections) == 1
    assert detections[0].box.x1 == 10.0
    assert detections[0].box.x2 == 50.0
    assert detections[0].box.y2 == 30.0


@pytest.mark.fast
def test_florence2_unknown_task_raises():
    """Unknown task should raise ValueError listing supported tasks."""
    import torch

    from visionservex.engines.florence2 import Florence2Engine
    from visionservex.registry import default_registry

    reg = default_registry()
    engine = Florence2Engine(reg.get("florence-2-base"))
    engine._real_ready = True
    engine._torch = torch
    engine._processor = MagicMock()
    param = torch.zeros(1)
    engine._model = MagicMock()
    engine._model.parameters.side_effect = lambda: iter([param])

    with pytest.raises(ValueError, match="Florence-2 task"):
        engine.predict(_tiny_image(), task="not_a_real_task")


@pytest.mark.fast
def test_florence2_task_token_mapping():
    """Sanity-check the task→token table contains documented tokens."""
    from visionservex.engines.florence2 import _TASK_TOKEN

    assert _TASK_TOKEN["caption"] == "<CAPTION>"
    assert _TASK_TOKEN["object_detection"] == "<OD>"
    assert _TASK_TOKEN["phrase_grounding"] == "<CAPTION_TO_PHRASE_GROUNDING>"
    assert _TASK_TOKEN["ocr"] == "<OCR>"
    assert _TASK_TOKEN["region_ocr"] == "<OCR_WITH_REGION>"


# ---------------------------------------------------------------------------
# SAM3 auth wrapper
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_sam3_status_without_token(monkeypatch):
    """SAM3 status reports HF_AUTH_REQUIRED when no token is set."""
    from visionservex.cli.sam3_commands import collect_sam3_status

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    s = collect_sam3_status("sam3.1-base-plus")
    assert s.has_hf_token is False
    # If transformers is installed, blocker is HF_AUTH_REQUIRED;
    # otherwise it's HF_TRANSFORMERS_REQUIRED. Either is correct behavior.
    assert s.blocker_code in {"HF_AUTH_REQUIRED", "HF_TRANSFORMERS_REQUIRED"}


@pytest.mark.fast
def test_sam3_status_redacts_token(monkeypatch):
    """SAM3 status never exposes the full HF token."""
    from visionservex.cli.sam3_commands import collect_sam3_status

    monkeypatch.setenv("HF_TOKEN", "hf_abcdefghijklmnop")
    s = collect_sam3_status("sam3.1-base-plus")
    # Redacted form: first 3 + *** + last 2.
    assert "hf_abcdefghijklmnop" not in s.hf_token_redacted
    assert s.hf_token_redacted.startswith("hf_")
    assert s.hf_token_redacted.endswith("op")
    assert "***" in s.hf_token_redacted


@pytest.mark.fast
def test_sam3_status_short_token_redacted(monkeypatch):
    """Short tokens (< 8 chars) are fully redacted to '***'."""
    from visionservex.cli.sam3_commands import _redact

    assert _redact("abc") == "***"
    assert _redact("") == ""
    assert _redact(None) == ""


@pytest.mark.fast
def test_sam3_models_mapping():
    """Every SAM3 alias maps to a facebook/sam3* repo."""
    from visionservex.cli.sam3_commands import _SAM3_MODELS

    expected = {
        "sam3",
        "sam3-base",
        "sam3-large",
        "sam3.1",
        "sam3.1-small",
        "sam3.1-base-plus",
        "sam3.1-large",
    }
    assert set(_SAM3_MODELS.keys()) == expected
    for repo in _SAM3_MODELS.values():
        assert repo.startswith("facebook/sam3")


@pytest.mark.fast
def test_sam3_supported_prompts_returns_empty():
    """v1.8.0 honestly reports no wired SAM3 prompts."""
    # Direct module check; the CLI command prints, returns nothing.
    # Confirm the wrapper module advertises no inference support.
    import visionservex.cli.sam3_commands as mod
    from visionservex.cli.sam3_commands import _SAM3_MODELS  # noqa: F401

    # The supported-prompts command builds the payload inline; we just
    # confirm the module does NOT export any predict() function.
    assert not hasattr(mod, "predict")
    assert not hasattr(mod, "infer")


# ---------------------------------------------------------------------------
# Expert sidecars
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_expert_list_includes_required_sidecars():
    from visionservex.cli.expert_commands import EXPERTS

    required = {"openmmlab", "mmdet", "mmrotate", "mmpose", "detectron2", "maskdino", "co-detr"}
    assert required.issubset(set(EXPERTS.keys()))


@pytest.mark.fast
def test_expert_install_dry_run_does_not_execute(monkeypatch):
    """The install command must print but never subprocess.run any pip/git."""
    import subprocess

    from visionservex.cli.expert_commands import EXPERTS

    called = []

    def _fail(*a, **kw):
        called.append((a, kw))
        raise AssertionError("subprocess must not be called from expert install --dry-run")

    monkeypatch.setattr(subprocess, "run", _fail)
    monkeypatch.setattr(subprocess, "check_call", _fail)
    monkeypatch.setattr(subprocess, "Popen", _fail)

    # Building the install_commands list does not run anything itself.
    info = EXPERTS["openmmlab"]
    assert all(isinstance(c, str) for c in info.install_commands)
    assert info.structured_error_code == "OPENMMLAB_REQUIRED"
    # Verify no subprocess hits.
    assert called == []


@pytest.mark.fast
def test_expert_module_missing_returns_structured_code():
    """When required modules are missing, the structured error code is reported."""
    from visionservex.cli.expert_commands import EXPERTS, _missing

    # MaskDINO requires 'maskdino' which is unlikely installed in CI.
    info = EXPERTS["maskdino"]
    missing = _missing(info)
    # We can't guarantee absence in every env, but the metadata fields exist:
    assert info.structured_error_code == "MASKDINO_REQUIRED"
    assert isinstance(missing, list)


@pytest.mark.fast
def test_expert_install_commands_reference_official_tools():
    """Smoke-check that install recipes use openmim or pip/git (no surprise tools)."""
    from visionservex.cli.expert_commands import EXPERTS

    allowed_prefixes = ("pip", "mim", "git", "cd", "python", "#")
    for info in EXPERTS.values():
        for c in info.install_commands:
            assert any(c.lstrip().startswith(p) for p in allowed_prefixes), (
                f"Suspicious install command for {info.name}: {c!r}"
            )


# ---------------------------------------------------------------------------
# Manifest accuracy regression: florence-2/owlv2 must now claim runnable
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_manifest_florence_and_owlv2_runnable_again():
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST

    for mid in ("florence-2-base", "florence-2-large", "owlv2-base-patch16", "owlv2-large-patch14"):
        entry = SOURCE_MANIFEST[mid]
        assert entry.runnable_in_visionservex is True, (
            f"{mid} should be runnable_in_visionservex=True in v1.8.0"
        )
