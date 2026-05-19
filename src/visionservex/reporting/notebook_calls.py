# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.39.0: notebook call ledger.

Tracks which models are actually invoked by an executable notebook cell,
not just mentioned in markdown. The ledger is the source of truth for
the ``called_in_notebook`` column on the reconciled coverage ledger.
"""

from __future__ import annotations

import csv as _csv
import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ALLOWED_CALL_TYPES: frozenset[str] = frozenset(
    {
        "benchmark",
        "contract",
        "smoke",
        "demo",
        "doctor",
        "status",
        "checkpoint_state",
        "auth_gate",
        "license_gate",
        "sidecar_status",
        "validator",
        "skipped_with_reason",
    }
)

ALLOWED_EXECUTION_STATUS: frozenset[str] = frozenset(
    {
        "executed",
        "executed_blocked",
        "skipped_external_auth",
        "skipped_manual_checkpoint",
        "skipped_license_opt_in",
        "skipped_deprecated",
        "skipped_wrong_registry_entry",
        "skipped_sidecar_unavailable",
    }
)

ALLOWED_SKIP_REASONS: frozenset[str] = frozenset(
    {
        "auth_required",
        "opt_in_license_required",
        "upstream_deprecated",
        "wrong_registry_entry",
        "manual_checkpoint_required",
        "sidecar_required",
    }
)


@dataclass
class NotebookCall:
    """One recorded invocation of a model from a notebook cell."""

    model_id: str
    family: str
    task: str
    notebook_path: str
    notebook_section: str
    call_type: str
    command_or_api: str
    called_in_notebook: bool
    call_count: int
    execution_status: str
    final_state: str = ""
    blocker_code: str = ""
    evidence_artifact: str = ""
    output_artifact_exists: bool = False
    run_id: str = ""
    timestamp: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class NotebookCallLedger:
    """In-memory + on-disk notebook call ledger."""

    path: Path
    run_id: str = ""
    calls: list[NotebookCall] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def init(cls, path: Path, run_id: str = "") -> NotebookCallLedger:
        rid = (
            run_id
            or os.environ.get("VISIONSERVEX_NOTEBOOK_RUN_ID")
            or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        ledger = cls(path=path, run_id=rid, calls=[])
        ledger.flush()
        return ledger

    @classmethod
    def load(cls, path: Path) -> NotebookCallLedger:
        if not path.exists():
            return cls.init(path)
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return cls.init(path)
        rid = payload.get("run_id", "")
        calls_raw = payload.get("calls", [])
        calls: list[NotebookCall] = []
        for c in calls_raw:
            extras = c.pop("extras", {}) if isinstance(c, dict) else {}
            try:
                calls.append(NotebookCall(extras=extras, **c))
            except TypeError:
                # tolerate missing/extra keys (schema evolution)
                allowed = NotebookCall.__dataclass_fields__.keys()
                clean = {k: c.get(k) for k in allowed if k in c}
                clean.setdefault("model_id", "")
                clean.setdefault("family", "")
                clean.setdefault("task", "")
                clean.setdefault("notebook_path", "")
                clean.setdefault("notebook_section", "")
                clean.setdefault("call_type", "status")
                clean.setdefault("command_or_api", "")
                clean.setdefault("called_in_notebook", False)
                clean.setdefault("call_count", 0)
                clean.setdefault("execution_status", "executed")
                calls.append(NotebookCall(extras=extras, **clean))
        return cls(path=path, run_id=rid, calls=calls)

    def add(self, call: NotebookCall) -> None:
        with self._lock:
            self.calls.append(call)
        self.flush()

    def flush(self) -> None:
        with self._lock:
            payload = {
                "run_id": self.run_id,
                "schema_version": 1,
                "total_calls": len(self.calls),
                "calls": [c.to_dict() for c in self.calls],
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self.path)

    def write_csv(self, csv_path: Path) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "model_id",
            "family",
            "task",
            "notebook_path",
            "notebook_section",
            "call_type",
            "command_or_api",
            "called_in_notebook",
            "call_count",
            "execution_status",
            "final_state",
            "blocker_code",
            "evidence_artifact",
            "output_artifact_exists",
            "run_id",
            "timestamp",
        ]
        with csv_path.open("w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for c in self.calls:
                row = {k: getattr(c, k) for k in fields}
                w.writerow(row)

    def iter_calls(self) -> Iterator[NotebookCall]:
        yield from self.calls

    def by_model(self) -> dict[str, list[NotebookCall]]:
        out: dict[str, list[NotebookCall]] = {}
        for c in self.calls:
            out.setdefault(c.model_id, []).append(c)
        return out

    def coverage_summary(self) -> dict[str, Any]:
        by_model = self.by_model()
        called = [m for m, cs in by_model.items() if any(c.called_in_notebook for c in cs)]
        skipped = [m for m, cs in by_model.items() if all(not c.called_in_notebook for c in cs)]
        return {
            "total_models": len(by_model),
            "called_in_notebook": len(called),
            "skipped": len(skipped),
            "called_model_ids": sorted(called),
            "skipped_model_ids": sorted(skipped),
            "run_id": self.run_id,
        }


_DEFAULT_LEDGER_PATH = Path(
    os.environ.get(
        "VISIONSERVEX_NOTEBOOK_CALL_LEDGER",
        str(Path.cwd() / "notebook/99_final_report/reports/notebook_model_call_ledger.json"),
    )
)


def _resolve_ledger(ledger: NotebookCallLedger | None) -> NotebookCallLedger:
    if ledger is not None:
        return ledger
    path = _DEFAULT_LEDGER_PATH
    if path.exists():
        return NotebookCallLedger.load(path)
    return NotebookCallLedger.init(path)


def record_model_call(
    *,
    model_id: str,
    notebook: str,
    section: str,
    task: str,
    command: str,
    call_type: str,
    status: str,
    final_state: str = "",
    blocker_code: str = "",
    evidence_artifact: str = "",
    output_artifact_exists: bool | None = None,
    family: str = "",
    extras: dict[str, Any] | None = None,
    ledger: NotebookCallLedger | None = None,
) -> NotebookCall:
    """Record one model invocation into the notebook call ledger.

    ``call_type`` must come from :data:`ALLOWED_CALL_TYPES`.
    ``status`` must come from :data:`ALLOWED_EXECUTION_STATUS`.
    """
    if call_type not in ALLOWED_CALL_TYPES:
        raise ValueError(
            f"call_type {call_type!r} not in ALLOWED_CALL_TYPES (got {sorted(ALLOWED_CALL_TYPES)})"
        )
    if status not in ALLOWED_EXECUTION_STATUS:
        raise ValueError(
            f"execution_status {status!r} not in ALLOWED_EXECUTION_STATUS "
            f"(got {sorted(ALLOWED_EXECUTION_STATUS)})"
        )
    led = _resolve_ledger(ledger)
    existed = output_artifact_exists
    if existed is None and evidence_artifact:
        p = Path(evidence_artifact)
        if p.is_absolute() and p.exists():
            existed = True
        else:
            # Try common notebook roots
            cands = [
                p,
                Path.cwd() / p,
                led.path.parent.parent.parent / p,  # NB root from <nb>/99_final_report/reports/
                led.path.parent.parent / p,
                Path.cwd() / "notebook" / p,
            ]
            existed = any(c.exists() for c in cands)
    call = NotebookCall(
        model_id=model_id,
        family=family,
        task=task,
        notebook_path=notebook,
        notebook_section=section,
        call_type=call_type,
        command_or_api=command,
        called_in_notebook=call_type != "skipped_with_reason",
        call_count=1,
        execution_status=status,
        final_state=final_state,
        blocker_code=blocker_code,
        evidence_artifact=evidence_artifact,
        output_artifact_exists=bool(existed),
        run_id=led.run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        extras=extras or {},
    )
    led.add(call)
    return call


def record_skip(
    *,
    model_id: str,
    notebook: str,
    section: str,
    task: str,
    reason: str,
    family: str = "",
    blocker_code: str = "",
    final_state: str = "",
    evidence_artifact: str = "",
    ledger: NotebookCallLedger | None = None,
) -> NotebookCall:
    """Record a precise allowed skip for a model that wasn't run in this notebook.

    ``reason`` must be one of :data:`ALLOWED_SKIP_REASONS`.
    """
    if reason not in ALLOWED_SKIP_REASONS:
        raise ValueError(
            f"skip reason {reason!r} not in ALLOWED_SKIP_REASONS (got {sorted(ALLOWED_SKIP_REASONS)})"
        )
    mapping = {
        "auth_required": "skipped_external_auth",
        "opt_in_license_required": "skipped_license_opt_in",
        "upstream_deprecated": "skipped_deprecated",
        "wrong_registry_entry": "skipped_wrong_registry_entry",
        "manual_checkpoint_required": "skipped_manual_checkpoint",
        "sidecar_required": "skipped_sidecar_unavailable",
    }
    return record_model_call(
        model_id=model_id,
        notebook=notebook,
        section=section,
        task=task,
        command=f"<skipped: {reason}>",
        call_type="skipped_with_reason",
        status=mapping[reason],
        final_state=final_state or reason,
        blocker_code=blocker_code or reason.upper(),
        evidence_artifact=evidence_artifact,
        family=family,
        extras={"skip_reason": reason},
        ledger=ledger,
    )


__all__ = [
    "ALLOWED_CALL_TYPES",
    "ALLOWED_EXECUTION_STATUS",
    "ALLOWED_SKIP_REASONS",
    "NotebookCall",
    "NotebookCallLedger",
    "record_model_call",
    "record_skip",
]
