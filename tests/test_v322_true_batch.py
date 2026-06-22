# SPDX-License-Identifier: Apache-2.0
"""v3.22.0 — true tensor batch honesty tests.

These tests FAIL if any engine advertises ``supports_true_batch=True`` while
actually looping over single-image predict (a hidden internal loop). They are
weight-free and run without torch/GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from visionservex.core.results import BaseResult, Box, Detection, DetectionResult
from visionservex.engines.base import BaseEngine
from visionservex.runtime.batch_infer import verify_true_forward_batch


class _FakeModel:
    """Stand-in torch module: counts forward calls."""

    def forward(self, batch: Any) -> Any:
        return [0] * (len(batch) if hasattr(batch, "__len__") else 1)


def _det(model_id: str = "fake") -> DetectionResult:
    return DetectionResult(
        kind="detection",
        model_id=model_id,
        detections=[Detection(box=Box(0, 0, 1, 1), score=0.9, label="x", class_id=0)],
    )


class _BaseFake(BaseEngine):
    def __init__(self) -> None:  # bypass entry requirement
        self.device = "cpu"
        self.precision = "fp32"
        self._loaded = True
        self._model = _FakeModel()

    def load(self, *, device: str, precision: str) -> None:  # pragma: no cover
        pass

    def infer(self, preprocessed: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return preprocessed

    def postprocess(self, raw: Any, *, image: Any, **kwargs: Any) -> BaseResult:  # pragma: no cover
        return _det()

    def predict(
        self, image: Any, *, prompts: Sequence[str] | None = None, **kwargs: Any
    ) -> BaseResult:
        # single-image path always calls forward once
        self._model.forward([image])
        return _det()


class HonestTrueBatchEngine(_BaseFake):
    """Genuine: ONE forward over the stacked batch."""

    supports_true_batch = True

    def predict_batch(
        self, images: Sequence[Any], *, prompts: Any = None, **kwargs: Any
    ) -> list[BaseResult]:
        self._model.forward(list(images))  # exactly one forward
        out = []
        for _ in images:
            r = _det()
            r.metadata.update(
                {
                    "batch_mode": "true_tensor_batch",
                    "true_forward_batch": True,
                    "internal_loop": False,
                }
            )
            out.append(r)
        return out


class LyingEngine(_BaseFake):
    """Claims true batch but inherits the BASE loop → forward called N times."""

    supports_true_batch = True
    # NOTE: does NOT override predict_batch → uses BaseEngine's honest loop.


class HonestLoopEngine(_BaseFake):
    """Honest single-image engine; default loop, no true-batch claim."""

    supports_true_batch = False


def test_verifier_confirms_real_true_batch() -> None:
    eng = HonestTrueBatchEngine()
    v = verify_true_forward_batch(eng, [object(), object(), object(), object()])
    assert v["forward_calls"] == 1
    assert v["is_true_forward_batch"] is True
    assert v["claim_matches_reality"] is True


def test_verifier_catches_lying_engine() -> None:
    """The whole point: an engine that claims true batch but loops is caught."""
    eng = LyingEngine()
    v = verify_true_forward_batch(eng, [object(), object(), object(), object()])
    assert v["forward_calls"] == 4, "lying engine looped → forward called per image"
    assert v["is_true_forward_batch"] is False
    assert v["claim_matches_reality"] is False  # claimed True, reality False


def test_default_batch_path_tags_internal_loop() -> None:
    eng = HonestLoopEngine()
    results = eng.predict_batch([object(), object()])
    assert all(r.metadata.get("batch_mode") == "internal_loop" for r in results)
    assert all(r.metadata.get("true_forward_batch") is False for r in results)


def test_structural_contract_real_engines() -> None:
    """Any engine class declaring supports_true_batch=True MUST override predict_batch.

    Catches a real engine that flips the flag without implementing a real batch.
    """
    # import engine modules so subclasses are registered
    import visionservex.engines  # noqa: F401

    offenders = []

    def walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            # only audit real shipped engines, not test fixtures
            if (
                sub.__module__.startswith("visionservex")
                and getattr(sub, "supports_true_batch", False)
                and sub.predict_batch is BaseEngine.predict_batch
            ):
                offenders.append(sub.__name__)
            walk(sub)

    walk(BaseEngine)
    assert not offenders, (
        f"engines claim supports_true_batch=True but use the base loop: {offenders}"
    )


def test_dfine_engine_declares_true_batch_and_overrides() -> None:
    from visionservex.engines.dfine import DFINEEngine

    assert DFINEEngine.supports_true_batch is True
    assert DFINEEngine.predict_batch is not BaseEngine.predict_batch
    assert DFINEEngine.max_batch_size_hint > 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
