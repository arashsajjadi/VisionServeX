"""Shared pytest fixtures, marker skip hooks, and resource guard integration.

Marker policy (all heavy markers are skipped by default):
  fast          unit test, no model load, no network, no GPU, no sidecar
  integration   CLI/registry/manifest integration without heavy resources
  slow          test takes > 2s; opt in: VISIONSERVEX_RUN_REAL_MODEL_TESTS=1
  real_model    requires real model weights; opt in: VISIONSERVEX_RUN_REAL_MODEL_TESTS=1
  gpu           requires a GPU device; opt in: VISIONSERVEX_RUN_GPU_TESTS=1
  network       requires live network; opt in: VISIONSERVEX_RUN_NETWORK_TESTS=1
  sidecar       requires heavy sidecar (Docker, OpenMMLab); opt in: VISIONSERVEX_RUN_SIDECAR_TESTS=1
  release       only during release validation
  benchmark     requires VISIONSERVEX_RUN_BENCHMARK_TESTS=1
  memory        reserved for memory-intensive tests; opt in: VISIONSERVEX_RUN_DISK_HEAVY_TESTS=1
  disk_heavy    reserved for disk-intensive tests; opt in: VISIONSERVEX_RUN_DISK_HEAVY_TESTS=1
  download      downloads large assets; opt in: VISIONSERVEX_RUN_DOWNLOAD_TESTS=1

Backward-compat aliases (old VISION_SERVEX_* prefix still accepted):
  VISION_SERVEX_RUN_REAL_MODEL_TESTS=1  →  enables real_model + slow
  VISION_SERVEX_RUN_GPU_TESTS=1         →  enables gpu
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Env-flag helpers — accept both naming conventions
# ---------------------------------------------------------------------------


def _flag(*names: str) -> bool:
    return any(os.environ.get(n, "0") == "1" for n in names)


_RUN_REAL_MODEL = _flag(
    "VISIONSERVEX_RUN_REAL_MODEL_TESTS",
    "VISION_SERVEX_RUN_REAL_MODEL_TESTS",
)
_RUN_GPU = _flag(
    "VISIONSERVEX_RUN_GPU_TESTS",
    "VISION_SERVEX_RUN_GPU_TESTS",
)
_RUN_NETWORK = _flag(
    "VISIONSERVEX_RUN_NETWORK_TESTS",
    "VISION_SERVEX_RUN_NETWORK_TESTS",
)
_RUN_SIDECAR = _flag(
    "VISIONSERVEX_RUN_SIDECAR_TESTS",
    "VISION_SERVEX_RUN_SIDECAR_TESTS",
)
_RUN_BENCHMARK = _flag("VISIONSERVEX_RUN_BENCHMARK_TESTS")
_RUN_DISK_HEAVY = _flag("VISIONSERVEX_RUN_DISK_HEAVY_TESTS")
_RUN_DOWNLOAD = _flag("VISIONSERVEX_RUN_DOWNLOAD_TESTS")


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register all custom markers so --strict-markers stays happy."""
    _markers = [
        ("fast", "unit test — no model load, no network, no GPU, no sidecar"),
        ("integration", "CLI/registry/manifest integration without heavy resources"),
        (
            "slow",
            "test takes more than ~2s; opt in: VISIONSERVEX_RUN_REAL_MODEL_TESTS=1",
        ),
        (
            "real_model",
            "requires real model weights; opt in: VISIONSERVEX_RUN_REAL_MODEL_TESTS=1",
        ),
        (
            "gpu",
            "requires a GPU device; opt in: VISIONSERVEX_RUN_GPU_TESTS=1",
        ),
        (
            "network",
            "requires live network; opt in: VISIONSERVEX_RUN_NETWORK_TESTS=1",
        ),
        (
            "sidecar",
            "requires heavy sidecar (Docker, OpenMMLab, Detectron2); "
            "opt in: VISIONSERVEX_RUN_SIDECAR_TESTS=1",
        ),
        ("release", "only run as part of release validation"),
        ("benchmark", "benchmark test; opt in: VISIONSERVEX_RUN_BENCHMARK_TESTS=1"),
        (
            "memory",
            "memory-intensive test; opt in: VISIONSERVEX_RUN_DISK_HEAVY_TESTS=1",
        ),
        (
            "disk_heavy",
            "disk-intensive test; opt in: VISIONSERVEX_RUN_DISK_HEAVY_TESTS=1",
        ),
        (
            "download",
            "downloads large assets; opt in: VISIONSERVEX_RUN_DOWNLOAD_TESTS=1",
        ),
        (
            "smoke",
            "smallest/lightest variant of a real_model or gpu test; always paired with real_model or gpu",
        ),
    ]
    for name, desc in _markers:
        config.addinivalue_line("markers", f"{name}: {desc}")


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------

_MARKER_SKIP_RULES: list[tuple[str, bool, str]] = [
    # (marker_name, enabled, skip_reason)
    (
        "real_model",
        _RUN_REAL_MODEL,
        "real model weights not available; set VISIONSERVEX_RUN_REAL_MODEL_TESTS=1",
    ),
    (
        "slow",
        _RUN_REAL_MODEL,
        "slow test disabled by default; set VISIONSERVEX_RUN_REAL_MODEL_TESTS=1",
    ),
    (
        "gpu",
        _RUN_GPU,
        "GPU not enabled; set VISIONSERVEX_RUN_GPU_TESTS=1 with a healthy GPU",
    ),
    (
        "network",
        _RUN_NETWORK,
        "network tests disabled by default; set VISIONSERVEX_RUN_NETWORK_TESTS=1",
    ),
    (
        "sidecar",
        _RUN_SIDECAR,
        "sidecar tests disabled by default; set VISIONSERVEX_RUN_SIDECAR_TESTS=1",
    ),
    (
        "benchmark",
        _RUN_BENCHMARK,
        "benchmark tests disabled by default; set VISIONSERVEX_RUN_BENCHMARK_TESTS=1",
    ),
    (
        "memory",
        _RUN_DISK_HEAVY,
        "memory-intensive tests disabled by default; set VISIONSERVEX_RUN_DISK_HEAVY_TESTS=1",
    ),
    (
        "disk_heavy",
        _RUN_DISK_HEAVY,
        "disk-heavy tests disabled by default; set VISIONSERVEX_RUN_DISK_HEAVY_TESTS=1",
    ),
    (
        "download",
        _RUN_DOWNLOAD,
        "download tests disabled by default; set VISIONSERVEX_RUN_DOWNLOAD_TESTS=1",
    ),
]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply skip markers based on enabled flags."""
    for marker_name, enabled, reason in _MARKER_SKIP_RULES:
        if enabled:
            continue
        skip_mark = pytest.mark.skip(reason=reason)
        for item in items:
            if marker_name in item.keywords:
                item.add_marker(skip_mark)


# ---------------------------------------------------------------------------
# Session-level resource guard: lockfile + concurrent-pytest detection
# ---------------------------------------------------------------------------


def pytest_sessionstart(session: pytest.Session) -> None:
    """Acquire pytest lock and check for concurrent test runs."""
    allow_concurrent = os.environ.get("VISIONSERVEX_ALLOW_CONCURRENT_PYTEST", "0") == "1"
    if allow_concurrent:
        return
    try:
        from visionservex.runtime.resource_guard import (
            ResourceGuardError,
            acquire_pytest_lock,
            refuse_if_other_pytest_running,
        )

        refuse_if_other_pytest_running()
        acquire_pytest_lock()
    except ImportError:
        pass  # resource_guard not importable (broken environment) — soft fail
    except ResourceGuardError as exc:
        pytest.exit(str(exc), returncode=3)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Release pytest lock and print resource summary."""
    try:
        from visionservex.runtime.resource_guard import (
            print_resource_report,
            release_pytest_lock,
        )

        release_pytest_lock()
        # Only print summary for non-trivial sessions (at least one test ran)
        if session.testscollected > 0:
            print_resource_report("session-end")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Per-test cleanup for heavy markers
# ---------------------------------------------------------------------------

_HEAVY_MARKERS = {"real_model", "gpu", "benchmark", "memory", "disk_heavy", "download"}


@pytest.fixture(autouse=True)
def _cleanup_heavy_test(request: pytest.FixtureRequest):
    """After each heavy test, run GC and CUDA cache flush."""
    yield
    is_heavy = any(m in request.node.keywords for m in _HEAVY_MARKERS)
    if not is_heavy:
        return
    try:
        from visionservex.runtime.resource_guard import cleanup_after_test

        cleanup_after_test()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Environment isolation fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Strip VISIONSERVEX_* / VISION_SERVEX_* env vars and use a temp cache."""
    for key in list(os.environ):
        if key.startswith("VISIONSERVEX_") or key.startswith("VISION_SERVEX_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path / "cache"))
    from visionservex.config import reload_settings

    reload_settings()
    yield
    reload_settings()


# ---------------------------------------------------------------------------
# Common image fixtures
# ---------------------------------------------------------------------------


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
