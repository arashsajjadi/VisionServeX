# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Pick a model for the user's hardware + task + difficulty preferences.

The recommender is intentionally simple and explainable. It returns a small
list of candidates ranked by:

1. Implementation status (`wired` > `partial` > `stub`).
2. Project status (`stable` > `beta` > `experimental` > others).
3. Beginner-friendliness if requested (``simple=True``).
4. Whether the model can plausibly run on the user's device/VRAM.
5. Difficulty (easier first when ``simple=True``).

Returns dataclasses so callers can format CLI/JSON output uniformly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from visionservex.registry import ModelEntry, Task, default_registry
from visionservex.runtime.device import DeviceInfo, available_devices, best_device


@dataclass
class Recommendation:
    entry: ModelEntry
    score: float
    reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "model_id": self.entry.id,
            "display_name": self.entry.display_name,
            "task": self.entry.task,
            "license": self.entry.license,
            "status": self.entry.status,
            "implementation_status": self.entry.implementation_status,
            "difficulty": self.entry.difficulty,
            "supported_devices": self.entry.supported_devices,
            "minimum_vram_gb": self.entry.minimum_vram_gb,
            "recommended_vram_gb": self.entry.recommended_vram_gb,
            "auto_download": self.entry.auto_download,
            "install_extra": self.entry.install_extra,
            "score": round(self.score, 3),
            "reasons": self.reasons,
        }


_DIFFICULTY_SCORE = {
    "very_easy": 5.0,
    "easy": 4.0,
    "medium": 3.0,
    "hard": 2.0,
    "expert": 1.0,
}

_STATUS_SCORE = {
    "stable": 5.0,
    "beta": 4.0,
    "experimental": 3.0,
    "optional": 2.5,
    "manual": 2.0,
    "external": 1.0,
    "stub": 1.0,
}

_IMPL_SCORE = {"wired": 5.0, "partial": 3.0, "stub": 1.0}


def _device_score(
    entry: ModelEntry, devices: list[DeviceInfo], vram_hint: float | None
) -> tuple[float, list[str]]:
    available = {d.name for d in devices if d.available}
    can_run = bool(set(entry.supported_devices) & available)
    reasons: list[str] = []
    score = 0.0
    if can_run:
        score += 3.0
        reasons.append(
            f"runs on available device(s): {sorted(set(entry.supported_devices) & available)}"
        )
    else:
        score -= 5.0
        reasons.append(f"none of supported devices {entry.supported_devices} are available")

    if vram_hint is not None and entry.minimum_vram_gb is not None:
        if vram_hint >= entry.minimum_vram_gb:
            score += 2.0
            reasons.append(f"VRAM hint {vram_hint:.1f} GB ≥ minimum {entry.minimum_vram_gb} GB")
            if entry.recommended_vram_gb and vram_hint >= entry.recommended_vram_gb:
                score += 1.0
                reasons.append(f"VRAM hint ≥ recommended {entry.recommended_vram_gb} GB")
        else:
            score -= 3.0
            reasons.append(f"VRAM hint {vram_hint:.1f} GB < minimum {entry.minimum_vram_gb} GB")
    return score, reasons


_GOAL_CATEGORY_BOOST: dict[str, dict[str, float]] = {
    "accuracy": {
        "accuracy_grade": 6.0,
        "production_recommended": 3.0,
        "experimental_sota": 1.0,
        "demo_fast": -4.0,
        "utility": -5.0,
        "expert_sidecar": -2.0,
    },
    "fastest_demo": {
        "demo_fast": 6.0,
        "production_recommended": 1.0,
        "accuracy_grade": -2.0,
        "utility": -5.0,
        "expert_sidecar": -3.0,
    },
    "best_open_license": {},
    "best_colab": {
        "demo_fast": 4.0,
        "production_recommended": 2.0,
        "accuracy_grade": 0.0,
        "expert_sidecar": -4.0,
    },
    "best_gpu": {
        "accuracy_grade": 4.0,
        "production_recommended": 2.0,
        "expert_sidecar": -2.0,
    },
    "best_cpu": {},
    "best_segmentation": {},
    "best_open_vocab": {},
}


def recommend(
    *,
    task: Task | str | None = None,
    device: str | None = None,
    vram_gb: float | None = None,
    simple: bool = False,
    goal: str | None = None,
    limit: int = 5,
    candidates: Iterable[ModelEntry] | None = None,
) -> list[Recommendation]:
    """Return up to ``limit`` recommendations.

    Args:
        goal: Optional recommendation goal. Values:
            accuracy, fastest_demo, best_open_license, best_colab,
            best_gpu, best_cpu, best_segmentation, best_open_vocab
    """
    entries = list(candidates) if candidates is not None else list(default_registry().list())

    # Goal-driven task remapping
    if goal == "best_segmentation" and task is None:
        task = "segment"
    elif goal == "best_open_vocab" and task is None:
        task = "open_vocab_detect"

    if task:
        entries = [e for e in entries if e.task == task]

    devices = available_devices()
    if device and device != "auto":
        entries = [e for e in entries if device.lower() in {d.lower() for d in e.supported_devices}]

    # Goal-driven CPU filter
    if goal == "best_cpu":
        entries = [e for e in entries if "cpu" in {d.lower() for d in e.supported_devices}]

    # Use user-provided VRAM hint, or derive from probe.
    effective_vram = vram_gb
    if effective_vram is None:
        best = best_device()
        effective_vram = best.total_vram_gb if best.total_vram_gb else None

    goal_boosts = _GOAL_CATEGORY_BOOST.get(goal or "", {})

    results: list[Recommendation] = []
    for entry in entries:
        reasons: list[str] = []
        score = 0.0

        impl = _IMPL_SCORE.get(entry.implementation_status, 1.0)
        score += impl
        reasons.append(f"implementation_status={entry.implementation_status} (+{impl})")

        st = _STATUS_SCORE.get(entry.status, 1.0)
        score += st
        reasons.append(f"status={entry.status} (+{st})")

        diff = _DIFFICULTY_SCORE.get(entry.difficulty, 3.0)
        score += diff if simple else diff * 0.5
        reasons.append(f"difficulty={entry.difficulty} (+{diff if simple else diff * 0.5:.1f})")

        if simple and entry.beginner_recommendation:
            score += 2.0
            reasons.append("flagged beginner_recommendation (+2.0)")

        if entry.auto_download:
            score += 1.0
            reasons.append("auto_download=true (+1.0)")

        dev_score, dev_reasons = _device_score(entry, devices, effective_vram)
        score += dev_score
        reasons.extend(dev_reasons)

        # Penalize manual/external/stub for simple recommendations.
        if simple:
            if entry.status in {"manual", "external"}:
                score -= 3.0
                reasons.append("penalty: simple mode prefers non-manual models (-3.0)")
            if entry.implementation_status == "stub":
                score -= 2.0
                reasons.append("penalty: simple mode penalizes stubs (-2.0)")

        # Goal-based category boost/penalty
        if goal and entry.model_category:
            boost = goal_boosts.get(entry.model_category, 0.0)
            if boost != 0.0:
                score += boost
                reasons.append(
                    f"goal={goal!r} category={entry.model_category} ({'+' if boost > 0 else ''}{boost})"
                )

        # Always penalize unavailable_with_reason for normal recommendations
        if entry.model_category == "unavailable_with_reason":
            score -= 8.0
            reasons.append("penalty: unavailable_with_reason (-8.0)")

        # Penalize experimental_sota unless specifically requested
        if entry.model_category == "experimental_sota" and goal not in ("accuracy",):
            score -= 4.0
            reasons.append("penalty: experimental_sota requires goal=accuracy to surface (-4.0)")

        results.append(Recommendation(entry=entry, score=score, reasons=reasons))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]


def first_beginner_pick(*, task: Task | str | None = None) -> ModelEntry | None:
    """Return the single best beginner pick. Used by `doctor`."""
    recs = recommend(task=task, simple=True, limit=1)
    return recs[0].entry if recs else None


__all__ = ["Recommendation", "first_beginner_pick", "recommend"]
