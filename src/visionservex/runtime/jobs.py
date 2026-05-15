# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""In-process job tracking for long-running operations (mainly model downloads).

Jobs are stable units of work that a server client can poll. They carry a
JSON-friendly state machine with these statuses:

    queued
    checking_dependencies
    downloading
    verifying
    loading_model
    running_inference
    completed
    failed
    cancelled

Stable progress envelope is defined in this module to keep server/CLI
contracts in sync.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

JobStatus = Literal[
    "queued",
    "checking_dependencies",
    "downloading",
    "verifying",
    "loading_model",
    "running_inference",
    "completed",
    "failed",
    "cancelled",
]


@dataclass
class Job:
    job_id: str
    model_id: str
    kind: str  # "pull", "predict", etc.
    status: JobStatus = "queued"
    message: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    progress: dict[str, Any] = field(default_factory=dict)
    result: Any | None = None
    error: dict[str, Any] | None = None
    cancelled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "kind": self.kind,
            "status": self.status,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "cancelled": self.cancelled,
        }


class JobStore:
    """Thread-safe in-process job registry."""

    def __init__(self, *, max_jobs: int = 256) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._max_jobs = max_jobs
        self._listeners: dict[str, list[Callable[[Job], None]]] = {}

    def create(self, *, model_id: str, kind: str = "pull") -> Job:
        with self._lock:
            self._gc()
            job_id = secrets.token_urlsafe(12)
            job = Job(job_id=job_id, model_id=model_id, kind=kind)
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        message: str | None = None,
        progress: dict[str, Any] | None = None,
        result: Any | None = None,
        error: dict[str, Any] | None = None,
    ) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if status is not None:
                job.status = status
            if message is not None:
                job.message = message
            if progress is not None:
                job.progress = progress
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            job.updated_at = time.time()
            for listener in self._listeners.get(job_id, []):
                try:
                    listener(job)
                except Exception:
                    pass
            return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status in {"completed", "failed", "cancelled"}:
                return False
            job.cancelled = True
            job.status = "cancelled"
            job.message = "cancelled by client"
            job.updated_at = time.time()
            return True

    def subscribe(self, job_id: str, listener: Callable[[Job], None]) -> None:
        with self._lock:
            self._listeners.setdefault(job_id, []).append(listener)

    def _gc(self) -> None:
        if len(self._jobs) <= self._max_jobs:
            return
        # Drop the oldest terminal jobs.
        terminal_statuses = {"completed", "failed", "cancelled"}
        sortable = sorted(
            (j for j in self._jobs.values() if j.status in terminal_statuses),
            key=lambda j: j.updated_at,
        )
        for old in sortable[: max(0, len(self._jobs) - self._max_jobs)]:
            self._jobs.pop(old.job_id, None)


_default_store: JobStore | None = None
_store_lock = threading.Lock()


def get_job_store() -> JobStore:
    global _default_store
    if _default_store is None:
        with _store_lock:
            if _default_store is None:
                _default_store = JobStore()
    return _default_store


__all__ = ["Job", "JobStatus", "JobStore", "get_job_store"]
