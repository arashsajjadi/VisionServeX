# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Phase 1 (v2.16.0): GPU profile classifier — RTX 5080 must not be t4_colab.

The notebook v16 run classified an RTX 5080 (15.99 GB VRAM, NVIDIA GeForce
RTX 5080) as profile=t4_colab because the old heuristic only looked at VRAM
size. ``classify_gpu`` now uses name + VRAM and must return one of the new
desktop_* profiles for consumer RTX cards.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

from visionservex.runtime.gpu_profile import (
    GPU_PROFILES,
    classify_gpu,
    detect_gpu_profile,
)


def test_rtx_5080_is_not_t4_colab() -> None:
    """The exact notebook regression: RTX 5080 must classify as a desktop profile."""
    profile, notes = classify_gpu("NVIDIA GeForce RTX 5080", 15.99)
    assert profile != "t4_colab", (
        f"RTX 5080 must not be classified as t4_colab. Got: {profile} notes={notes}"
    )
    assert profile == "desktop_16gb_fast", profile


def test_rtx_4080_desktop_16gb() -> None:
    profile, _ = classify_gpu("NVIDIA GeForce RTX 4080", 15.7)
    assert profile == "desktop_16gb_fast"


def test_rtx_4090_desktop_24gb() -> None:
    profile, _ = classify_gpu("NVIDIA GeForce RTX 4090", 23.6)
    assert profile == "desktop_24gb_fast"


def test_rtx_3090_desktop_24gb() -> None:
    profile, _ = classify_gpu("NVIDIA GeForce RTX 3090", 23.5)
    assert profile == "desktop_24gb_fast"


def test_t4_colab_by_name() -> None:
    profile, _ = classify_gpu("Tesla T4", 15.0)
    assert profile == "t4_colab"


def test_l4_colab_by_name() -> None:
    profile, _ = classify_gpu("NVIDIA L4", 22.5)
    assert profile == "l4_colab"


def test_a100_colab_by_name() -> None:
    profile, _ = classify_gpu("NVIDIA A100-SXM4-40GB", 39.6)
    assert profile == "a100_colab"


def test_h100_colab_by_name() -> None:
    profile, _ = classify_gpu("NVIDIA H100 80GB HBM3", 79.1)
    assert profile == "h100_colab"


def test_no_gpu_is_cpu_only() -> None:
    profile, _ = classify_gpu(None, None)
    assert profile == "cpu_only"


def test_rtx_a4000_workstation_not_l4() -> None:
    """A workstation RTX A4000 must not be mistaken for an L4 just because L4 is a substring."""
    profile, _ = classify_gpu("NVIDIA RTX A4000", 16.0)
    # We don't require a specific bucket here — the only requirement is
    # that it's NOT l4_colab and NOT t4_colab.
    assert profile not in ("l4_colab", "t4_colab"), profile


def test_unknown_card_falls_back_by_vram() -> None:
    profile, _ = classify_gpu("Acme Quantum Card", 80.0)
    assert profile == "desktop_32gb_plus"


def test_all_profiles_are_in_GPU_PROFILES() -> None:
    """Every classification path must produce a valid profile name."""
    cases = [
        (None, None),
        ("Tesla T4", 14.7),
        ("NVIDIA L4", 22.5),
        ("NVIDIA A100", 40.0),
        ("NVIDIA H100", 80.0),
        ("NVIDIA GeForce RTX 5080", 16.0),
        ("NVIDIA GeForce RTX 4090", 24.0),
        ("Mystery Card", 8.0),
        ("NVIDIA RTX A4000", 16.0),
    ]
    for name, vram in cases:
        profile, _ = classify_gpu(name, vram)
        assert profile in GPU_PROFILES, profile


def test_detect_gpu_profile_does_not_raise() -> None:
    """``detect_gpu_profile`` is safe to call when torch isn't available."""
    p = detect_gpu_profile()
    assert p.profile in GPU_PROFILES
    assert isinstance(p.to_dict(), dict)


def test_dev_gpu_profile_cli(tmp_path) -> None:
    """The `visionservex dev gpu-profile --format json --out` CLI shape."""
    binary = shutil.which("visionservex")
    binary_args = [sys.executable, "-m", "visionservex"] if binary is None else [binary]
    out = tmp_path / "gpu_profile.json"
    res = subprocess.run(
        [*binary_args, "dev", "gpu-profile", "--format", "json", "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert out.exists()
    payload = json.loads(out.read_text())
    assert "profile" in payload
    assert payload["profile"] in GPU_PROFILES
    assert "cuda_available" in payload
    assert "recommended_small_workers" in payload
