"""Scheduler and model cache tests."""

from __future__ import annotations

import asyncio

import pytest

from visionservex.runtime.cache import ModelCache
from visionservex.runtime.scheduler import (
    BackpressureError,
    RequestScheduler,
)


@pytest.mark.asyncio
async def test_scheduler_serializes_within_limit():
    sched = RequestScheduler(per_model_concurrency=2, queue_size=5, request_timeout_s=2.0)

    counter = {"running": 0, "peak": 0}

    async def work():
        counter["running"] += 1
        counter["peak"] = max(counter["peak"], counter["running"])
        await asyncio.sleep(0.05)
        counter["running"] -= 1

    async with sched.reserve("m1"):
        await work()
    assert counter["peak"] == 1


@pytest.mark.asyncio
async def test_scheduler_backpressure():
    sched = RequestScheduler(per_model_concurrency=1, queue_size=1, request_timeout_s=2.0)

    async def hold():
        async with sched.reserve("m"):
            await asyncio.sleep(0.2)

    async def fail_fast():
        with pytest.raises(BackpressureError):
            async with sched.reserve("m"):
                pass

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0.01)
    await fail_fast()
    await holder


def test_cache_lru_eviction():
    cache = ModelCache(max_loaded=2)
    cache.get("mock-detect")
    cache.get("mock-segment")
    cache.get("mock-pose")  # should evict mock-detect
    keys = cache.keys()
    assert "mock-detect" not in keys
    assert "mock-segment" in keys
    assert "mock-pose" in keys
    cache.clear()


def test_cache_reuses_same_instance():
    cache = ModelCache(max_loaded=2)
    a = cache.get("mock-detect")
    b = cache.get("mock-detect")
    assert a is b
    cache.clear()
