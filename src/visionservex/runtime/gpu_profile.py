# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""GPU profile classification.

The notebook benchmark misclassified an RTX 5080 as ``t4_colab`` because the
old heuristic only looked at VRAM size. This module classifies GPUs by both
name and VRAM so that a 16 GB consumer RTX is not bucketed as a 16 GB T4.

Profiles
--------
- ``cpu_only``         — no CUDA visible at all.
- ``t4_colab``         — NVIDIA T4 (free-tier Colab).
- ``l4_colab``         — NVIDIA L4 (Colab Pro tier).
- ``a100_colab``       — NVIDIA A100 (Colab Pro+/research).
- ``h100_colab``       — NVIDIA H100 (datacentre).
- ``desktop_16gb_fast`` -- consumer RTX with ~12-20 GB VRAM (e.g. RTX 4080/5080).
- ``desktop_24gb_fast`` -- consumer RTX with ~20-28 GB VRAM (RTX 3090/4090/5090).
- ``desktop_32gb_plus`` -- workstation card with >28 GB VRAM.
- ``unknown_cuda``     — CUDA present, but the card couldn't be classified.

The classifier is deliberately conservative: name matches win over VRAM, so a
T4 reporting an odd VRAM number still ends up as ``t4_colab``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "GPU_PROFILES",
    "GpuProfile",
    "classify_gpu",
    "detect_gpu_profile",
]

GPU_PROFILES = (
    "cpu_only",
    "t4_colab",
    "l4_colab",
    "a100_colab",
    "h100_colab",
    "desktop_16gb_fast",
    "desktop_24gb_fast",
    "desktop_32gb_plus",
    "unknown_cuda",
)


@dataclass
class GpuProfile:
    """Result of GPU profile classification."""

    cuda_available: bool
    gpu_name: str
    total_vram_gb: float
    profile: str
    recommended_small_workers: int
    recommended_medium_workers: int
    recommended_heavy_workers: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "total_vram_gb": round(self.total_vram_gb, 2),
            "profile": self.profile,
            "recommended_small_workers": self.recommended_small_workers,
            "recommended_medium_workers": self.recommended_medium_workers,
            "recommended_heavy_workers": self.recommended_heavy_workers,
            "notes": list(self.notes),
        }


# Consumer RTX detection: match RTX followed by a 4-digit model number.
# Covers RTX 30xx / 40xx / 50xx and any future 60xx series.
_RTX_CONSUMER_RE = re.compile(r"\bRTX\s*(\d{3,4})\b", re.IGNORECASE)


def _workers_for(profile: str) -> tuple[int, int, int]:
    """Return (small, medium, heavy) recommended concurrent model counts."""
    if profile in ("h100_colab", "desktop_32gb_plus"):
        return 4, 2, 1
    if profile in ("a100_colab", "desktop_24gb_fast"):
        return 3, 2, 1
    if profile in ("desktop_16gb_fast", "l4_colab"):
        return 2, 1, 1
    if profile == "t4_colab":
        return 1, 1, 1
    if profile == "unknown_cuda":
        return 1, 1, 1
    return 1, 1, 1  # cpu_only


def classify_gpu(name: str | None, total_vram_gb: float | None) -> tuple[str, list[str]]:
    """Classify a GPU into one of :data:`GPU_PROFILES`.

    Name matching wins over VRAM. ``None`` for both returns ``cpu_only``.
    """
    notes: list[str] = []
    if not name and not total_vram_gb:
        return "cpu_only", ["no GPU detected"]

    n = (name or "").strip()
    n_lower = n.lower()
    vram = float(total_vram_gb or 0.0)

    # Datacentre cards — name-driven.
    if "h100" in n_lower:
        return "h100_colab", notes
    if "a100" in n_lower:
        return "a100_colab", notes
    if "l4" in n_lower and "rtx" not in n_lower:
        # "NVIDIA L4" without RTX prefix → Colab Pro L4. "RTX A4000" must not match.
        return "l4_colab", notes
    if re.search(r"\bt4\b", n_lower):
        return "t4_colab", notes

    # Consumer RTX — name + VRAM.
    rtx_match = _RTX_CONSUMER_RE.search(n)
    if rtx_match:
        model = int(rtx_match.group(1))
        notes.append(f"matched consumer RTX {model}")
        # 90-class typically 24 GB; 80-class 16 GB; 70 12 GB.
        if vram > 28.0:
            return "desktop_32gb_plus", notes
        if vram >= 20.0:
            return "desktop_24gb_fast", notes
        if vram >= 11.0:
            return "desktop_16gb_fast", notes
        # < 11 GB -- still a consumer RTX, classify by VRAM band closest to 16 GB band.
        return "desktop_16gb_fast", [
            *notes,
            "VRAM lower than typical 16GB band; using desktop_16gb_fast",
        ]

    # Generic VRAM-based fallback if we know nothing.
    vram_note = "unknown name; classified by VRAM"
    if vram > 28.0:
        return "desktop_32gb_plus", [*notes, vram_note]
    if vram >= 20.0:
        return "desktop_24gb_fast", [*notes, vram_note]
    if vram >= 13.0:
        return "desktop_16gb_fast", [*notes, vram_note]

    return "unknown_cuda", [*notes, f"could not classify GPU name {n!r}; vram={vram:.1f}GB"]


def detect_gpu_profile() -> GpuProfile:
    """Probe the real environment and return a :class:`GpuProfile`.

    Uses ``torch.cuda`` when available, otherwise returns ``cpu_only``.
    Safe to call when torch is not installed.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return _cpu_only_profile()

    try:
        if not torch.cuda.is_available():
            return _cpu_only_profile()
    except Exception:
        return _cpu_only_profile()

    try:
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_vram_gb = props.total_memory / (1024**3)
    except Exception as exc:
        notes = [f"CUDA visible but probing failed: {exc!s:.120}"]
        return GpuProfile(
            cuda_available=True,
            gpu_name="unknown",
            total_vram_gb=0.0,
            profile="unknown_cuda",
            recommended_small_workers=1,
            recommended_medium_workers=1,
            recommended_heavy_workers=1,
            notes=notes,
        )

    profile, notes = classify_gpu(name, total_vram_gb)
    small, medium, heavy = _workers_for(profile)
    return GpuProfile(
        cuda_available=True,
        gpu_name=name,
        total_vram_gb=total_vram_gb,
        profile=profile,
        recommended_small_workers=small,
        recommended_medium_workers=medium,
        recommended_heavy_workers=heavy,
        notes=notes,
    )


def _cpu_only_profile() -> GpuProfile:
    return GpuProfile(
        cuda_available=False,
        gpu_name="",
        total_vram_gb=0.0,
        profile="cpu_only",
        recommended_small_workers=1,
        recommended_medium_workers=1,
        recommended_heavy_workers=1,
        notes=["no CUDA available"],
    )
