# SPDX-License-Identifier: Apache-2.0
"""v3.18 typed public API: wrong-task calls raise, never silently mis-route.

Weight-free — mock engines only.
"""

from __future__ import annotations

import pytest
from PIL import Image

from visionservex import VisionModel
from visionservex.exceptions import TaskNotSupportedError

_IMG = Image.new("RGB", (48, 48), (10, 20, 30))


def test_classify_on_detector_raises_typed_error():
    m = VisionModel("mock-detect")
    with pytest.raises(TaskNotSupportedError) as ei:
        m.classify(_IMG)
    assert ei.value.code == "TASK_NOT_SUPPORTED"


def test_detect_on_classifier_raises_typed_error():
    m = VisionModel("mock-classify")
    with pytest.raises(TaskNotSupportedError) as ei:
        m.detect(_IMG)
    assert ei.value.code == "TASK_NOT_SUPPORTED"


def test_embed_on_detector_raises_typed_error():
    with pytest.raises(TaskNotSupportedError):
        VisionModel("mock-detect").embed(_IMG)


def test_segment_on_classifier_raises_typed_error():
    with pytest.raises(TaskNotSupportedError):
        VisionModel("mock-classify").segment(_IMG)


def test_correspond_always_points_to_insid3():
    with pytest.raises(TaskNotSupportedError) as ei:
        VisionModel("mock-detect").correspond(_IMG, _IMG)
    assert ei.value.code == "TASK_NOT_SUPPORTED"
    assert "insid3" in (ei.value.hint or "").lower()


def test_detect_on_open_vocab_is_allowed():
    # open_vocab_detect IS a detection task -> detect() must NOT raise.
    r = VisionModel("mock-open-vocab").detect(_IMG, prompts=["thing"])
    assert r is not None


def test_classify_on_classifier_is_allowed():
    r = VisionModel("mock-classify").classify(_IMG)
    assert r is not None
