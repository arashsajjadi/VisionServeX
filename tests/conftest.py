"""Shared pytest fixtures.

We isolate every test from the user's environment by:

* clearing VISIONSERVEX_* env vars (overrides leak between processes otherwise);
* pointing the cache dir at a temp directory;
* reloading settings before each test.
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
