# SPDX-License-Identifier: Apache-2.0
"""Benchmark smoke tests — opt-in via VISIONSERVEX_RUN_BENCHMARK_TESTS=1.

All benchmarks here are:
  - Bounded to ≤ 3 images
  - Process-isolated (each model load is scoped)
  - Artifacts go to tmp_path only
  - Resource-guarded before starting
"""

from __future__ import annotations

import time

import pytest
from PIL import Image


def _tiny_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color=(100, 100, 100))


@pytest.mark.benchmark
@pytest.mark.smoke
def test_mock_detect_benchmark_timing(tmp_path):
    """Benchmark smoke: mock-detect model timing, max 3 runs, output to tmp_path."""
    from visionservex import VisionModel
    from visionservex.runtime.resource_guard import (
        ResourceGuardError,
        assert_safe_to_start_benchmark,
    )

    try:
        assert_safe_to_start_benchmark()
    except ResourceGuardError as exc:
        pytest.skip(f"Resource guard: {exc}")

    img = _tiny_image()
    timings = []

    with VisionModel("mock-detect", device="cpu") as m:
        for _ in range(3):
            t0 = time.perf_counter()
            m.predict(img)
            timings.append((time.perf_counter() - t0) * 1000)

    assert len(timings) == 3
    assert all(t >= 0 for t in timings), "Timings must be non-negative"

    out = tmp_path / "benchmark_result.json"
    import json

    out.write_text(json.dumps({"model": "mock-detect", "timings_ms": timings}))
    assert out.exists()
    assert out.stat().st_size < 10 * 1024, "Benchmark output must be < 10 KB"


@pytest.mark.benchmark
@pytest.mark.smoke
def test_mock_segment_benchmark(tmp_path):
    """Benchmark smoke: mock-segment, max 3 runs."""
    from visionservex import VisionModel
    from visionservex.runtime.resource_guard import (
        ResourceGuardError,
        assert_safe_to_start_benchmark,
    )

    try:
        assert_safe_to_start_benchmark()
    except ResourceGuardError as exc:
        pytest.skip(f"Resource guard: {exc}")

    img = _tiny_image()
    with VisionModel("mock-segment", device="cpu") as m:
        for _ in range(3):
            result = m.predict(img)

    assert result is not None


@pytest.mark.benchmark
@pytest.mark.smoke
@pytest.mark.real_model
def test_real_model_benchmark_smoke(tmp_path):
    """Real model benchmark smoke: dfine-s, 3 images only. Skips if checkpoint missing."""
    from visionservex import VisionModel
    from visionservex.runtime.resource_guard import (
        ResourceGuardError,
        assert_safe_to_start_model_load,
        cleanup_after_test,
    )

    try:
        assert_safe_to_start_model_load(required_vram_gb=1.0)
    except ResourceGuardError as exc:
        pytest.skip(f"Resource guard: {exc}")

    img = _tiny_image()
    timings = []

    try:
        with VisionModel("dfine-s-o365-coco", device="cpu") as m:
            for _ in range(3):
                t0 = time.perf_counter()
                m.predict(img)
                timings.append((time.perf_counter() - t0) * 1000)
    except Exception as exc:
        msg = str(exc)
        if "checkpoint" in msg.lower() or "download" in msg.lower():
            pytest.skip(f"Checkpoint not available: {exc}")
        raise
    finally:
        cleanup_after_test()

    assert len(timings) == 3
    import json

    (tmp_path / "dfine_s_bench.json").write_text(
        json.dumps({"model": "dfine-s-o365-coco", "timings_ms": timings})
    )
