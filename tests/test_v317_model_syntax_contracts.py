# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.17.0: task-specific public API routes correctly and raises typed errors."""

from __future__ import annotations

import numpy as np
import pytest

from visionservex.core.model import VisionModel
from visionservex.exceptions import TaskNotSupportedError


def test_methods_exist():
    for m in ("classify", "embed", "segment", "similarity", "correspond"):
        assert callable(getattr(VisionModel, m, None))


def test_classify_wrong_task_raises():
    with pytest.raises(TaskNotSupportedError) as e:
        VisionModel("libreyolo-yolox-s").classify("x.jpg")  # detector, not classify
    assert "TASK_NOT_SUPPORTED" in str(e.value)


def test_embed_wrong_task_raises():
    with pytest.raises(TaskNotSupportedError):
        VisionModel("torchvision-resnet50").embed("x.jpg")  # classifier, not embed


def test_segment_wrong_task_raises():
    with pytest.raises(TaskNotSupportedError):
        VisionModel("torchvision-resnet50").segment("x.jpg")


def test_correspond_always_typed_error():
    with pytest.raises(TaskNotSupportedError) as e:
        VisionModel("dinov2-base").correspond("a.jpg", "b.jpg")
    assert "INSID3" in str(e.value)


def test_similarity_is_cosine():
    m = VisionModel("dinov2-base")
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert abs(m.similarity(a, b) - 1.0) < 1e-5
    assert abs(m.similarity(a, c)) < 1e-5


def test_similarity_accepts_embedding_result():
    from visionservex.core.embedding_results import EmbeddingResult

    m = VisionModel("dinov2-base")
    ea = EmbeddingResult(model_id="x", embedding=np.array([1.0, 1.0], dtype=np.float32))
    eb = EmbeddingResult(model_id="x", embedding=np.array([1.0, 1.0], dtype=np.float32))
    assert abs(m.similarity(ea, eb) - 1.0) < 1e-5


@pytest.mark.parametrize(
    "mid", ["libreyolo-yolox-s", "torchvision-resnet50", "dinov2-base", "sam2-hiera-small"]
)
def test_validated_syntax_present(mid):
    syn = VisionModel(mid).capabilities()["validated_syntax"]
    assert "predict" in syn and all(isinstance(v, str) for v in syn.values())
