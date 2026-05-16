# SPDX-License-Identifier: Apache-2.0
"""Tests for resource_guard.py — all mocked; no real memory/VRAM is consumed."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import guard — resource_guard must be importable without torch
# ---------------------------------------------------------------------------


def _guard():
    from visionservex.runtime import resource_guard

    return resource_guard


@pytest.mark.fast
def test_resource_guard_imports():
    rg = _guard()
    assert hasattr(rg, "assert_safe_to_start_test")
    assert hasattr(rg, "get_system_memory_state")
    assert hasattr(rg, "get_gpu_memory_state")
    assert hasattr(rg, "get_disk_state")
    assert hasattr(rg, "ResourceGuardError")


# ---------------------------------------------------------------------------
# get_system_memory_state
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_get_system_memory_state_returns_values():
    rg = _guard()
    state = rg.get_system_memory_state()
    assert state.total_gb > 0
    assert 0 <= state.used_pct <= 100
    assert state.available_gb >= 0


# ---------------------------------------------------------------------------
# get_disk_state
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_get_disk_state_returns_values(tmp_path):
    rg = _guard()
    d = rg.get_disk_state(tmp_path)
    assert d.total_gb > 0
    assert d.free_gb >= 0


# ---------------------------------------------------------------------------
# get_gpu_memory_state — mocked (no real GPU required)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_get_gpu_memory_state_no_cuda():
    """When torch is missing, should return zeroed state."""
    rg = _guard()
    with patch.dict("sys.modules", {"torch": None}):
        state = rg.get_gpu_memory_state()
    assert not state.cuda_available
    assert state.free_gb == 0.0


@pytest.mark.fast
def test_resource_guard_refuses_high_ram_mocked(monkeypatch):
    """refuse_if_ram_above_threshold raises when RAM is over the limit."""
    rg = _guard()

    fake_mem = MagicMock()
    fake_mem.used_pct = 95.0
    fake_mem.available_gb = 1.0
    fake_mem.total_gb = 64.0

    monkeypatch.setattr(rg, "get_system_memory_state", lambda: fake_mem)
    monkeypatch.setenv("VISIONSERVEX_ALLOW_HEAVY_TESTS", "0")
    # Re-read the constant — monkeypatch the module-level flag
    monkeypatch.setattr(rg, "ALLOW_HEAVY", False)

    with pytest.raises(rg.ResourceGuardError, match="RAM usage"):
        rg.refuse_if_ram_above_threshold(max_pct=80.0)


@pytest.mark.fast
def test_resource_guard_refuses_high_vram_mocked(monkeypatch):
    """refuse_if_vram_above_threshold raises when free VRAM is insufficient."""
    rg = _guard()

    fake_gpu = MagicMock()
    fake_gpu.cuda_available = True
    fake_gpu.free_gb = 0.5
    fake_gpu.device_name = "RTX 5080 (mock)"

    monkeypatch.setattr(rg, "get_gpu_memory_state", lambda: fake_gpu)
    monkeypatch.setattr(rg, "MIN_FREE_VRAM_GB", 2.0)
    monkeypatch.setattr(rg, "ALLOW_HEAVY", False)

    with pytest.raises(rg.ResourceGuardError, match="VRAM"):
        rg.refuse_if_vram_above_threshold(required_vram_gb=4.0)


@pytest.mark.fast
def test_resource_guard_refuses_low_disk_mocked(monkeypatch, tmp_path):
    """refuse_if_disk_free_below_threshold raises when disk is too full."""
    rg = _guard()

    fake_disk = MagicMock()
    fake_disk.free_gb = 1.0

    monkeypatch.setattr(rg, "get_disk_state", lambda path="/": fake_disk)
    monkeypatch.setattr(rg, "MIN_FREE_DISK_GB", 10.0)
    monkeypatch.setattr(rg, "ALLOW_HEAVY", False)

    with pytest.raises(rg.ResourceGuardError, match="disk"):
        rg.refuse_if_disk_free_below_threshold(path=str(tmp_path))


# ---------------------------------------------------------------------------
# Pytest lock
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_pytest_lock_refuses_concurrent_run_mocked(monkeypatch, tmp_path):
    """acquire_pytest_lock raises if a lock exists with a live PID."""
    rg = _guard()

    lock_path = tmp_path / "test.lock"
    lock_data = {"pid": os.getpid(), "command": "pytest", "started": time.time(), "repo": "/tmp"}
    lock_path.write_text(json.dumps(lock_data))

    monkeypatch.setattr(rg, "PYTEST_LOCK_PATH", lock_path)
    monkeypatch.setattr(rg, "ALLOW_CONCURRENT_PYTEST", False)

    with pytest.raises(rg.ResourceGuardError, match="already active"):
        rg.acquire_pytest_lock()


@pytest.mark.fast
def test_pytest_lock_cleans_stale(monkeypatch, tmp_path):
    """acquire_pytest_lock replaces a lock whose PID no longer exists."""
    rg = _guard()

    lock_path = tmp_path / "test.lock"
    # PID 99999999 almost certainly doesn't exist
    lock_path.write_text(json.dumps({"pid": 99999999, "command": "pytest", "started": 0}))

    monkeypatch.setattr(rg, "PYTEST_LOCK_PATH", lock_path)
    monkeypatch.setattr(rg, "ALLOW_CONCURRENT_PYTEST", False)

    rg.acquire_pytest_lock()  # must not raise
    data = json.loads(lock_path.read_text())
    assert data["pid"] == os.getpid()

    # cleanup
    rg.release_pytest_lock()
    assert not lock_path.exists()


@pytest.mark.fast
def test_release_pytest_lock_only_own_pid(monkeypatch, tmp_path):
    """release_pytest_lock does not remove a lock owned by a different PID."""
    rg = _guard()

    lock_path = tmp_path / "test.lock"
    other_pid = os.getpid() + 1  # different PID
    lock_path.write_text(json.dumps({"pid": other_pid, "command": "pytest", "started": 0}))

    monkeypatch.setattr(rg, "PYTEST_LOCK_PATH", lock_path)
    rg.release_pytest_lock()
    # Should still exist because pid doesn't match
    assert lock_path.exists()


# ---------------------------------------------------------------------------
# Process guard
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_refuse_if_other_pytest_running_passes_when_none(monkeypatch):
    """refuse_if_other_pytest_running does not raise when no other pytest is running."""
    rg = _guard()

    monkeypatch.setattr(rg, "get_process_tree", lambda: [])
    monkeypatch.setattr(rg, "ALLOW_CONCURRENT_PYTEST", False)

    rg.refuse_if_other_pytest_running()  # must not raise


@pytest.mark.fast
def test_refuse_if_other_pytest_running_raises(monkeypatch):
    rg = _guard()

    fake_proc = MagicMock()
    fake_proc.is_pytest = True
    fake_proc.pid = 99998

    monkeypatch.setattr(rg, "get_process_tree", lambda: [fake_proc])
    monkeypatch.setattr(rg, "ALLOW_CONCURRENT_PYTEST", False)

    with pytest.raises(rg.ResourceGuardError, match="already active"):
        rg.refuse_if_other_pytest_running()


# ---------------------------------------------------------------------------
# ResourceBudget.to_dict
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_resource_budget_to_dict():
    rg = _guard()
    budget = rg._collect_resource_budget()
    d = budget.to_dict()
    assert "ram_available_gb" in d
    assert "vram_free_gb" in d
    assert "disk_free_gb" in d
    assert "cpu_pct" in d


# ---------------------------------------------------------------------------
# cleanup_after_test — must not raise even without torch
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_cleanup_after_test_no_torch():
    rg = _guard()
    with patch.dict("sys.modules", {"torch": None}):
        rg.cleanup_after_test()  # must not raise


# ---------------------------------------------------------------------------
# kill_visionservex_tests — returns a list (no real kill in test)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_kill_visionservex_tests_returns_list(monkeypatch):
    rg = _guard()
    # Patch process iteration to return nothing relevant
    monkeypatch.setattr(
        "psutil.process_iter",
        lambda attrs=None: iter([]),
    )
    result = rg.kill_visionservex_tests()
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Artifact size guard (sanity check — no real files written)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_artifact_size_guard(tmp_path):
    """Ensure large fake artifacts are detectable by checking file size."""
    large_file = tmp_path / "huge.parquet"
    # Write 1 MB fake artifact
    large_file.write_bytes(b"x" * (1024 * 1024))
    assert large_file.stat().st_size > 512 * 1024


# ---------------------------------------------------------------------------
# No generated artifacts in repo root
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_no_outputs_reports_indexes_committed():
    """Generated output dirs should not contain tracked files (git-unaware check)."""
    repo = Path(__file__).resolve().parent.parent
    suspicious_exts = {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".engine",
        ".plan",
        ".trt",
        ".bin",
        ".h5",
        ".pb",
        ".tflite",
        ".npy",
        ".parquet",
        ".faiss",
    }
    forbidden_dirs = {"outputs", "reports", "indexes", "runs"}
    for d in forbidden_dirs:
        target = repo / d
        if not target.exists():
            continue
        for f in target.rglob("*"):
            if f.is_file() and f.suffix in suspicious_exts:
                pytest.fail(
                    f"Large generated artifact found in repo: {f}. "
                    "Add it to .gitignore and remove it from the tree."
                )
