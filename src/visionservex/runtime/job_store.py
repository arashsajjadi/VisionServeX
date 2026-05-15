# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Optional SQLite-backed job store.

The default store is the in-process JobStore from ``jobs.py``.
When ``VISIONSERVEX_JOBS__STORE=sqlite`` is set, a SQLite-backed store is used
instead, which persists jobs across server restarts.

Config:
    VISIONSERVEX_JOBS__STORE       memory | sqlite   (default: memory)
    VISIONSERVEX_JOBS__SQLITE_PATH .visionservex/jobs.db
    VISIONSERVEX_JOBS__RETENTION_HOURS  24 (0 = never purge)
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from visionservex.runtime.jobs import Job, JobStatus
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)

_DEFAULT_DB = Path.home() / ".visionservex" / "jobs.db"


class SQLiteJobStore:
    """Thread-safe SQLite-backed job store with TTL-based cleanup."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        retention_hours: float = 24.0,
    ) -> None:
        self._path = Path(
            db_path or os.environ.get("VISIONSERVEX_JOBS__SQLITE_PATH", str(_DEFAULT_DB))
        )
        self._retention_hours = float(
            os.environ.get("VISIONSERVEX_JOBS__RETENTION_HOURS", str(retention_hours))
        )
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id      TEXT PRIMARY KEY,
                    model_id    TEXT NOT NULL,
                    kind        TEXT NOT NULL DEFAULT 'predict',
                    status      TEXT NOT NULL DEFAULT 'queued',
                    message     TEXT NOT NULL DEFAULT '',
                    progress    TEXT NOT NULL DEFAULT '{}',
                    result      TEXT,
                    error       TEXT,
                    cancelled   INTEGER NOT NULL DEFAULT 0,
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL
                )
            """)
            c.commit()

    # ---- conversion ----

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        j = Job(
            job_id=row["job_id"],
            model_id=row["model_id"],
            kind=row["kind"],
        )
        j.status = row["status"]
        j.message = row["message"]
        j.progress = json.loads(row["progress"])
        j.result = json.loads(row["result"]) if row["result"] else None
        j.error = json.loads(row["error"]) if row["error"] else None
        j.cancelled = bool(row["cancelled"])
        j.created_at = row["created_at"]
        j.updated_at = row["updated_at"]
        return j

    # ---- public API (mirrors JobStore) ----

    def create(self, *, model_id: str, kind: str = "pull") -> Job:
        import secrets

        with self._lock:
            job_id = secrets.token_urlsafe(12)
            now = time.time()
            j = Job(job_id=job_id, model_id=model_id, kind=kind)
            j.created_at = now  # type: ignore[assignment]
            j.updated_at = now  # type: ignore[assignment]
            with self._conn() as c:
                c.execute(
                    "INSERT INTO jobs (job_id, model_id, kind, status, message, progress, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (job_id, model_id, kind, "queued", "", "{}", now, now),
                )
                c.commit()
            return j

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            with self._conn() as c:
                row = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return None
            return self._row_to_job(row)

    def list(self) -> list[Job]:
        with self._lock:
            with self._conn() as c:
                rows = c.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
            return [self._row_to_job(r) for r in rows]

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
            j = self.get(job_id)
            if j is None:
                return None
            now = time.time()
            if status is not None:
                j.status = status
            if message is not None:
                j.message = message
            if progress is not None:
                j.progress = progress
            if result is not None:
                j.result = result
            if error is not None:
                j.error = error
            j.updated_at = now  # type: ignore[assignment]
            with self._conn() as c:
                c.execute(
                    "UPDATE jobs SET status=?,message=?,progress=?,result=?,error=?,updated_at=?"
                    " WHERE job_id=?",
                    (
                        j.status,
                        j.message,
                        json.dumps(j.progress),
                        json.dumps(j.result) if j.result is not None else None,
                        json.dumps(j.error) if j.error is not None else None,
                        now,
                        job_id,
                    ),
                )
                c.commit()
            return j

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            j = self.get(job_id)
            if j is None:
                return False
            terminal = {"completed", "failed", "cancelled"}
            if j.status in terminal:
                return False
            now = time.time()
            with self._conn() as c:
                c.execute(
                    "UPDATE jobs SET status='cancelled', cancelled=1, message='cancelled by client', updated_at=?"
                    " WHERE job_id=?",
                    (now, job_id),
                )
                c.commit()
            return True

    def purge_old(self) -> int:
        """Delete terminal jobs older than retention_hours. Returns count deleted."""
        if self._retention_hours <= 0:
            return 0
        cutoff = time.time() - self._retention_hours * 3600
        terminal = ("completed", "failed", "cancelled")
        with self._lock, self._conn() as c:
            result = c.execute(
                "DELETE FROM jobs WHERE status IN (?,?,?) AND updated_at < ?",
                (*terminal, cutoff),
            )
            c.commit()
            return result.rowcount

    def __len__(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]


def get_job_store_backend():
    """Return either the in-memory or SQLite job store depending on config."""
    mode = os.environ.get("VISIONSERVEX_JOBS__STORE", "memory").lower()
    if mode == "sqlite":
        _log.info(
            "Using SQLite job store at %s",
            os.environ.get("VISIONSERVEX_JOBS__SQLITE_PATH", str(_DEFAULT_DB)),
        )
        return SQLiteJobStore()
    from visionservex.runtime.jobs import get_job_store

    return get_job_store()


__all__ = ["SQLiteJobStore", "get_job_store_backend"]
