# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Adaptive GPU batch scheduler (v3.22.0).

A *measured* controller that picks the microbatch size for video / frame waves.
It targets high throughput **safely** — it does not blindly fill VRAM, and it
never claims the GPU is saturated; it reports the bottleneck it actually measured
(preprocess / forward / postprocess / vram / oom).

Design is deterministic and side-effect-free so it can be unit-tested by feeding
it synthetic :class:`WaveOutcome` records (``tests/test_v322_adaptive_scheduler.py``)
AND driven live by the video pipeline. The controller does NOT run inference
itself — callers run a wave at ``scheduler.batch_size`` then call
``scheduler.record(outcome)`` to get the next size + a human-readable reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Mode = Literal[
    "balanced",
    "max_throughput",
    "low_memory",
    "low_latency",
    "small_objects",
    "segmentation_quality",
]

# Non-power-of-two ladder (spec Phase 3.2). Model-specific caps clamp the top.
DEFAULT_LADDER: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)


@dataclass
class SchedulerConfig:
    vram_total_mb: float
    mode: Mode = "balanced"
    ladder: tuple[int, ...] = DEFAULT_LADDER
    model_max_batch: int = 128
    # VRAM fraction gates (spec Phase 3.5).
    target_vram_frac: float = 0.85
    hard_vram_frac: float = 0.92
    emergency_vram_frac: float = 0.95
    # Anti-thrash: require >this fractional throughput gain to grow a rung.
    grow_min_gain: float = 0.05
    # Treat a stage as the bottleneck if it exceeds this fraction of wall time.
    cpu_bound_frac: float = 0.55
    # Latency-spike guard for low_latency mode (ms per image ceiling).
    latency_ceiling_ms: float | None = None

    def start_batch(self) -> int:
        if self.mode == "low_memory":
            return 1
        if self.mode == "low_latency":
            return 1
        if self.mode == "max_throughput":
            return 4
        return 2

    def cap(self) -> int:
        return min(self.model_max_batch, self.ladder[-1])


@dataclass
class WaveOutcome:
    """What actually happened when a wave ran at ``batch_size``."""

    batch_size: int
    throughput_fps: float = 0.0
    latency_ms: float = 0.0  # wall time for the wave
    vram_used_mb: float = 0.0
    vram_free_mb: float = 0.0
    gpu_util_avg: float | None = None
    preprocess_ms: float = 0.0
    forward_ms: float = 0.0
    postprocess_ms: float = 0.0
    oom: bool = False
    cancel_requested: bool = False
    queue_depth: int = 0


@dataclass
class Decision:
    next_batch_size: int
    reason: str
    bottleneck: (
        str  # forward | preprocess | postprocess | vram | oom | cancel | converged | probing
    )
    action: str  # grow | shrink | hold | halt
    vram_frac: float = 0.0


class AdaptiveBatchScheduler:
    """Stateful microbatch controller with hysteresis + VRAM safety + OOM recovery."""

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self._ladder = [b for b in config.ladder if b <= config.cap()] or [1]
        self.batch_size = self._snap(config.start_batch())
        self._best_fps: float = 0.0
        self._best_batch: int = self.batch_size
        self._ceiling: int = config.cap()  # lowered when OOM/VRAM pressure seen
        self.history: list[Decision] = []
        self.converged: bool = False

    # ----- ladder helpers -----
    def _snap(self, b: int) -> int:
        """Snap an arbitrary size to the nearest ladder rung <= b (>=1)."""
        below = [x for x in self._ladder if x <= b]
        return below[-1] if below else self._ladder[0]

    def _rung_index(self, b: int) -> int:
        for i, x in enumerate(self._ladder):
            if x >= b:
                return i
        return len(self._ladder) - 1

    def _next_rung(self, b: int) -> int:
        i = self._rung_index(b)
        return self._ladder[min(i + 1, len(self._ladder) - 1)]

    def _prev_rung(self, b: int) -> int:
        i = self._rung_index(b)
        return self._ladder[max(i - 1, 0)]

    # ----- core decision -----
    def record(self, outcome: WaveOutcome) -> Decision:
        cfg = self.config
        vram_frac = outcome.vram_used_mb / cfg.vram_total_mb if cfg.vram_total_mb > 0 else 0.0

        # 1) Cancellation halts growth immediately (Phase 3.4).
        if outcome.cancel_requested:
            d = Decision(
                self.batch_size,
                "cancel requested — stop submitting new waves",
                "cancel",
                "halt",
                vram_frac,
            )
            return self._commit(d)

        # 2) OOM → halve (drop a couple of rungs) and set a hard ceiling (Phase 3.4/4).
        if outcome.oom:
            new = self._snap(max(1, outcome.batch_size // 2))
            self._ceiling = max(1, outcome.batch_size - 1)
            d = Decision(
                new,
                f"OOM at batch {outcome.batch_size} → halve to {new}; ceiling now {self._ceiling}",
                "oom",
                "shrink",
                vram_frac,
            )
            return self._commit(d)

        # 3) Emergency / hard VRAM stop (Phase 3.5).
        if vram_frac >= cfg.emergency_vram_frac:
            new = self._snap(max(1, outcome.batch_size // 2))
            self._ceiling = min(self._ceiling, max(1, outcome.batch_size - 1))
            d = Decision(
                new,
                f"VRAM {vram_frac:.0%} ≥ emergency {cfg.emergency_vram_frac:.0%} "
                f"→ emergency shrink to {new}",
                "vram",
                "shrink",
                vram_frac,
            )
            return self._commit(d)
        if vram_frac >= cfg.hard_vram_frac:
            new = self._prev_rung(outcome.batch_size)
            self._ceiling = min(self._ceiling, outcome.batch_size)
            d = Decision(
                new,
                f"VRAM {vram_frac:.0%} ≥ hard {cfg.hard_vram_frac:.0%} → shrink to {new}",
                "vram",
                "shrink",
                vram_frac,
            )
            return self._commit(d)

        # Track best throughput seen.
        if outcome.throughput_fps > self._best_fps:
            self._best_fps = outcome.throughput_fps
            self._best_batch = outcome.batch_size

        # 4) Bottleneck attribution (Phase 3 acceptance: surface measured bottleneck).
        cpu_ms = outcome.preprocess_ms + outcome.postprocess_ms
        total_stage = cpu_ms + outcome.forward_ms
        bottleneck = "forward"
        if total_stage > 0:
            if outcome.preprocess_ms / total_stage >= cfg.cpu_bound_frac:
                bottleneck = "preprocess"
            elif outcome.postprocess_ms / total_stage >= cfg.cpu_bound_frac:
                bottleneck = "postprocess"
            elif cpu_ms > outcome.forward_ms:
                bottleneck = (
                    "preprocess"
                    if outcome.preprocess_ms >= outcome.postprocess_ms
                    else "postprocess"
                )

        # 5) low_latency mode: respect per-image latency ceiling.
        if cfg.mode == "low_latency" and cfg.latency_ceiling_ms is not None:
            per_img = outcome.latency_ms / max(1, outcome.batch_size)
            if per_img > cfg.latency_ceiling_ms and outcome.batch_size > 1:
                new = self._prev_rung(outcome.batch_size)
                d = Decision(
                    new,
                    f"per-image latency {per_img:.0f}ms > ceiling "
                    f"{cfg.latency_ceiling_ms:.0f}ms → shrink",
                    "forward",
                    "shrink",
                    vram_frac,
                )
                return self._commit(d)

        # 6) Converged? At ceiling, or already explored a higher rung w/o gain.
        at_top = outcome.batch_size >= min(self._ceiling, self._ladder[-1])
        if at_top:
            self.converged = True
            d = Decision(
                outcome.batch_size,
                f"at batch ceiling {outcome.batch_size} (bottleneck={bottleneck}); holding",
                bottleneck,
                "hold",
                vram_frac,
            )
            return self._commit(d)

        # 7) CPU-bound and not improving → do NOT keep growing (Phase 3 acceptance).
        improving = outcome.throughput_fps >= self._best_fps * (1.0 + cfg.grow_min_gain)
        headroom_ok = vram_frac < cfg.target_vram_frac
        if bottleneck in ("preprocess", "postprocess") and not improving:
            self.converged = True
            d = Decision(
                outcome.batch_size,
                f"{bottleneck}-bound (cpu {cpu_ms:.0f}ms vs forward "
                f"{outcome.forward_ms:.0f}ms) and throughput plateaued → hold at "
                f"{outcome.batch_size}; growing batch won't help the GPU",
                bottleneck,
                "hold",
                vram_frac,
            )
            return self._commit(d)

        # 8) Grow if there's VRAM headroom and either improving or still probing.
        if headroom_ok and (improving or outcome.batch_size <= self._best_batch):
            new = self._next_rung(outcome.batch_size)
            new = min(new, self._ceiling)
            if new == outcome.batch_size:
                self.converged = True
                d = Decision(new, f"cannot grow past ceiling {new}", bottleneck, "hold", vram_frac)
            else:
                d = Decision(
                    new,
                    f"throughput {outcome.throughput_fps:.0f}fps, VRAM {vram_frac:.0%} "
                    f"< target {cfg.target_vram_frac:.0%} → probe up to {new}",
                    bottleneck,
                    "grow",
                    vram_frac,
                )
            return self._commit(d)

        # 9) Otherwise hold (stable region — anti-thrash).
        self.converged = True
        d = Decision(
            outcome.batch_size,
            f"stable at {outcome.batch_size} "
            f"(throughput {outcome.throughput_fps:.0f}fps, bottleneck={bottleneck})",
            bottleneck,
            "hold",
            vram_frac,
        )
        return self._commit(d)

    def _commit(self, d: Decision) -> Decision:
        self.batch_size = max(1, min(d.next_batch_size, self._ceiling, self._ladder[-1]))
        d.next_batch_size = self.batch_size
        self.history.append(d)
        return d


__all__ = [
    "DEFAULT_LADDER",
    "AdaptiveBatchScheduler",
    "Decision",
    "Mode",
    "SchedulerConfig",
    "WaveOutcome",
]
