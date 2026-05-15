# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Runtime: device selection, scheduling, model cache, monitoring, downloads, jobs."""

from visionservex.runtime.cache import ModelCache, get_model_cache
from visionservex.runtime.device import DeviceInfo, available_devices, best_device, resolve_device
from visionservex.runtime.jobs import Job, JobStatus, JobStore, get_job_store
from visionservex.runtime.monitor import MetricsRegistry, metrics
from visionservex.runtime.recommendations import Recommendation, first_beginner_pick, recommend
from visionservex.runtime.scheduler import RequestScheduler, get_scheduler

__all__ = [
    "DeviceInfo",
    "Job",
    "JobStatus",
    "JobStore",
    "MetricsRegistry",
    "ModelCache",
    "Recommendation",
    "RequestScheduler",
    "available_devices",
    "best_device",
    "first_beginner_pick",
    "get_job_store",
    "get_model_cache",
    "get_scheduler",
    "metrics",
    "recommend",
    "resolve_device",
]
