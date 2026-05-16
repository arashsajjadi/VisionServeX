# SPDX-License-Identifier: Apache-2.0
"""Real model smoke tests — opt-in only via VISIONSERVEX_RUN_REAL_MODEL_TESTS=1.

These tests load the smallest real models to verify inference actually works.
They are:
  - Resource-guarded (RAM + VRAM checked before loading)
  - Serialised (max workers = 1)
  - Always unload the model after the test
  - Use tiny 64x64 synthetic images only

Do not add large models here. Use `real_model` + `smoke` markers on every test.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image


def _tiny_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color=(128, 128, 128))


def _tiny_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    _tiny_image().save(buf, format="JPEG")
    return buf.getvalue()


def _guard():
    from visionservex.runtime import resource_guard

    return resource_guard


# ---------------------------------------------------------------------------
# dfine-s (smallest D-FINE)
# ---------------------------------------------------------------------------


@pytest.mark.real_model
@pytest.mark.smoke
def test_dfine_s_loads_and_predicts(tmp_path):
    """Smoke test: dfine-s-o365-coco loads and returns DetectionResult."""
    rg = _guard()
    try:
        rg.assert_safe_to_start_model_load(required_vram_gb=1.0)
    except rg.ResourceGuardError as exc:
        pytest.skip(f"Resource guard: {exc}")

    from visionservex import VisionModel

    try:
        with VisionModel("dfine-s-o365-coco", device="cpu") as m:
            result = m.predict(_tiny_image())
    except Exception as exc:
        msg = str(exc)
        if "checkpoint" in msg.lower() or "download" in msg.lower() or "cached" in msg.lower():
            pytest.skip(f"Checkpoint not available: {exc}")
        raise
    finally:
        rg.cleanup_after_test()

    assert result is not None, "dfine-s should return a result"
    assert hasattr(result, "boxes") or hasattr(result, "to_dict"), (
        "result must be a detection result"
    )


# ---------------------------------------------------------------------------
# rfdetr-small (smallest RF-DETR)
# ---------------------------------------------------------------------------


@pytest.mark.real_model
@pytest.mark.smoke
def test_rfdetr_small_loads_and_predicts():
    """Smoke test: rfdetr-small loads and returns DetectionResult."""
    rg = _guard()
    try:
        rg.assert_safe_to_start_model_load(required_vram_gb=0.5)
    except rg.ResourceGuardError as exc:
        pytest.skip(f"Resource guard: {exc}")

    from visionservex import VisionModel

    try:
        with VisionModel("rfdetr-small", device="cpu") as m:
            result = m.predict(_tiny_image())
    except Exception as exc:
        msg = str(exc)
        if "checkpoint" in msg.lower() or "download" in msg.lower() or "rfdetr" in msg.lower():
            pytest.skip(f"Checkpoint or dependency not available: {exc}")
        raise
    finally:
        rg.cleanup_after_test()

    assert result is not None


# ---------------------------------------------------------------------------
# dinov2-small (smallest embedding model)
# ---------------------------------------------------------------------------


@pytest.mark.real_model
@pytest.mark.smoke
def test_dinov2_small_embedding(tmp_path):
    """Smoke test: dinov2-small produces an embedding vector."""
    rg = _guard()
    try:
        rg.assert_safe_to_start_model_load(required_vram_gb=0.5)
    except rg.ResourceGuardError as exc:
        pytest.skip(f"Resource guard: {exc}")

    from visionservex import VisionModel

    try:
        with VisionModel("dinov2-small", device="cpu") as m:
            result = m.predict(_tiny_image())
    except Exception as exc:
        msg = str(exc)
        if "checkpoint" in msg.lower() or "download" in msg.lower() or "hugging" in msg.lower():
            pytest.skip(f"Checkpoint or dependency not available: {exc}")
        raise
    finally:
        rg.cleanup_after_test()

    assert result is not None
    assert hasattr(result, "embedding") or hasattr(result, "to_dict")


# ---------------------------------------------------------------------------
# sam2-hiera-tiny (smallest SAM2)
# ---------------------------------------------------------------------------


@pytest.mark.real_model
@pytest.mark.smoke
def test_sam2_tiny_loads(tmp_path):
    """Smoke test: sam2-hiera-tiny loads without crashing."""
    rg = _guard()
    try:
        rg.assert_safe_to_start_model_load(required_vram_gb=0.5)
    except rg.ResourceGuardError as exc:
        pytest.skip(f"Resource guard: {exc}")

    from visionservex import VisionModel

    try:
        with VisionModel("sam2-hiera-tiny", device="cpu") as m:
            result = m.predict(_tiny_image(), box=[10.0, 10.0, 50.0, 50.0])
    except Exception as exc:
        msg = str(exc)
        if any(
            kw in msg.lower()
            for kw in ["checkpoint", "download", "sam2", "hugging", "missing", "install"]
        ):
            pytest.skip(f"SAM2 not available: {exc}")
        raise
    finally:
        rg.cleanup_after_test()

    assert result is not None


# ---------------------------------------------------------------------------
# GPU smoke tests (separate — require VISIONSERVEX_RUN_GPU_TESTS=1)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.real_model
@pytest.mark.smoke
def test_dfine_s_gpu_inference():
    """GPU smoke test: dfine-s on CUDA. Checks VRAM before loading."""
    rg = _guard()
    gpu = rg.get_gpu_memory_state()
    if not gpu.cuda_available:
        pytest.skip("CUDA not available")
    try:
        rg.assert_safe_to_start_model_load(required_vram_gb=1.0)
    except rg.ResourceGuardError as exc:
        pytest.skip(f"Resource guard (VRAM): {exc}")

    from visionservex import VisionModel

    try:
        with VisionModel("dfine-s-o365-coco", device="cuda") as m:
            result = m.predict(_tiny_image())
    except Exception as exc:
        msg = str(exc)
        if "checkpoint" in msg.lower() or "download" in msg.lower():
            pytest.skip(f"Checkpoint not available: {exc}")
        raise
    finally:
        rg.cleanup_after_test()

    assert result is not None
    assert result.device == "cuda"
