# SPDX-License-Identifier: Apache-2.0
"""v3.22.0 — adaptive batch scheduler decision tests (deterministic, no GPU)."""

from __future__ import annotations

from visionservex.runtime.adaptive_batch import (
    AdaptiveBatchScheduler,
    SchedulerConfig,
    WaveOutcome,
)


def _sched(**kw) -> AdaptiveBatchScheduler:
    cfg = SchedulerConfig(vram_total_mb=16000.0, **kw)
    return AdaptiveBatchScheduler(cfg)


def test_probes_up_ladder_when_headroom_and_improving() -> None:
    s = _sched(mode="max_throughput")
    b0 = s.batch_size
    # GPU-bound, improving throughput, low VRAM → should grow.
    d = s.record(
        WaveOutcome(
            batch_size=b0,
            throughput_fps=100,
            vram_used_mb=1000,
            vram_free_mb=15000,
            forward_ms=30,
            preprocess_ms=2,
            postprocess_ms=1,
        )
    )
    assert d.action == "grow"
    assert d.next_batch_size > b0
    assert d.bottleneck == "forward"


def test_oom_halves_and_sets_ceiling() -> None:
    s = _sched()
    s.batch_size = 32
    d = s.record(WaveOutcome(batch_size=32, oom=True, vram_used_mb=15800, vram_free_mb=100))
    assert d.action == "shrink"
    assert d.bottleneck == "oom"
    assert d.next_batch_size <= 16
    # ceiling lowered → cannot grow back above 31
    d2 = s.record(
        WaveOutcome(
            batch_size=d.next_batch_size,
            throughput_fps=999,
            vram_used_mb=1000,
            vram_free_mb=15000,
            forward_ms=30,
        )
    )
    assert d2.next_batch_size <= 31


def test_emergency_vram_shrinks() -> None:
    s = _sched()
    s.batch_size = 16
    d = s.record(
        WaveOutcome(
            batch_size=16, throughput_fps=200, vram_used_mb=15300, vram_free_mb=700, forward_ms=20
        )
    )
    assert d.action == "shrink"
    assert d.bottleneck == "vram"


def test_cpu_bound_plateau_holds_not_grow() -> None:
    """The owner's exact scenario: preprocess dominates, GPU idle → DON'T keep growing."""
    s = _sched(mode="max_throughput")
    # establish a best throughput
    s.record(
        WaveOutcome(
            batch_size=4,
            throughput_fps=200,
            vram_used_mb=500,
            vram_free_mb=15000,
            forward_ms=10,
            preprocess_ms=2,
            postprocess_ms=1,
        )
    )
    # now CPU-preprocess-bound and NOT improving
    d = s.record(
        WaveOutcome(
            batch_size=8,
            throughput_fps=205,
            vram_used_mb=800,
            vram_free_mb=15000,
            forward_ms=13,
            preprocess_ms=40,
            postprocess_ms=2,
        )
    )
    assert d.action == "hold"
    assert d.bottleneck == "preprocess"
    assert "won't help" in d.reason


def test_cancel_halts() -> None:
    s = _sched()
    d = s.record(WaveOutcome(batch_size=8, cancel_requested=True))
    assert d.action == "halt"
    assert d.bottleneck == "cancel"


def test_converges_and_holds_at_ceiling() -> None:
    s = _sched(mode="max_throughput", model_max_batch=8)
    # keep feeding improving outcomes; must stop at model cap 8
    last = None
    for _ in range(10):
        last = s.record(
            WaveOutcome(
                batch_size=s.batch_size,
                throughput_fps=100 + s.batch_size,
                vram_used_mb=1000,
                vram_free_mb=15000,
                forward_ms=30,
            )
        )
    assert s.batch_size <= 8
    assert last is not None and last.action in ("hold",)


def test_low_latency_respects_ceiling() -> None:
    s = _sched(mode="low_latency", latency_ceiling_ms=20.0)
    s.batch_size = 8
    d = s.record(
        WaveOutcome(
            batch_size=8,
            throughput_fps=100,
            latency_ms=400,
            vram_used_mb=1000,
            vram_free_mb=15000,
            forward_ms=50,
        )
    )
    # 400ms/8 = 50ms per image > 20ms ceiling → shrink
    assert d.action == "shrink"


def test_decision_history_records_reasons() -> None:
    s = _sched()
    s.record(
        WaveOutcome(
            batch_size=2, throughput_fps=100, vram_used_mb=500, vram_free_mb=15000, forward_ms=30
        )
    )
    assert s.history
    assert all(d.reason for d in s.history)
