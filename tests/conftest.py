"""Shared pytest fixtures and marker skip hooks.

We isolate every test from the user's environment by:

* clearing VISIONSERVEX_* env vars (overrides leak between processes otherwise);
* pointing the cache dir at a temp directory;
* reloading settings before each test.

Marker policy
-------------
``@pytest.mark.real_model``: requires real model weights; skipped by default in CI.
``@pytest.mark.gpu``:        requires a healthy CUDA/MPS device; skipped if none.
``@pytest.mark.slow``:       runs take more than ~10 s; skipped by default in CI.

Set ``VISION_SERVEX_RUN_REAL_MODEL_TESTS=1`` to enable real_model and slow tests.
Set ``VISION_SERVEX_RUN_GPU_TESTS=1`` to enable GPU-only tests (implies a healthy GPU).
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

# Ensure the src layout is importable for tests.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_RUN_REAL_MODEL = os.environ.get("VISION_SERVEX_RUN_REAL_MODEL_TESTS", "0") == "1"
_RUN_GPU = os.environ.get("VISION_SERVEX_RUN_GPU_TESTS", "0") == "1"


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers so ``--strict-markers`` stays happy."""
    config.addinivalue_line(
        "markers",
        "real_model: test requires real model weights (skipped by default; "
        "set VISION_SERVEX_RUN_REAL_MODEL_TESTS=1 to enable)",
    )
    config.addinivalue_line(
        "markers",
        "gpu: test requires a healthy CUDA or MPS device "
        "(set VISION_SERVEX_RUN_GPU_TESTS=1 to enable)",
    )
    config.addinivalue_line(
        "markers",
        "slow: test takes more than ~10 s (skipped by default; "
        "set VISION_SERVEX_RUN_REAL_MODEL_TESTS=1 to enable)",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip real_model, gpu, and slow tests unless the corresponding env var is set."""
    skip_real = pytest.mark.skip(
        reason="real model weights not available in CI; "
        "set VISION_SERVEX_RUN_REAL_MODEL_TESTS=1 to enable"
    )
    skip_gpu = pytest.mark.skip(
        reason="GPU not available or not enabled; "
        "set VISION_SERVEX_RUN_GPU_TESTS=1 with a healthy GPU to enable"
    )
    skip_slow = pytest.mark.skip(
        reason="slow test disabled by default; set VISION_SERVEX_RUN_REAL_MODEL_TESTS=1 to enable"
    )

    for item in items:
        if "real_model" in item.keywords and not _RUN_REAL_MODEL:
            item.add_marker(skip_real)
        if "gpu" in item.keywords and not _RUN_GPU:
            item.add_marker(skip_gpu)
        if "slow" in item.keywords and not _RUN_REAL_MODEL:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Strip VISIONSERVEX_* and VISION_SERVEX_* env vars and use a temp cache."""
    for key in list(os.environ):
        if key.startswith("VISIONSERVEX_") or key.startswith("VISION_SERVEX_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path / "cache"))
    from visionservex.config import reload_settings

    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def jpeg_bytes() -> bytes:
    img = Image.new("RGB", (320, 240), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def big_png_bytes() -> bytes:
    img = Image.new("RGB", (200, 200), color="green")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
