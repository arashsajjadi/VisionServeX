# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.39.0: notebook-side helper that records model calls into the ledger.

Importable from any task notebook via:

    sys.path.insert(0, str(NB_ROOT / "shared"))
    from notebook_call_tracker import track_model_call, record_model_call_simple

Sends every model invocation to the canonical ledger at
``notebook/99_final_report/reports/notebook_model_call_ledger.json``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Resolve the canonical ledger path even when the working directory shifts.
_DEFAULT_NB_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LEDGER_PATH = _DEFAULT_NB_ROOT / "99_final_report/reports/notebook_model_call_ledger.json"


def _resolve_ledger_path() -> Path:
    env = os.environ.get("VISIONSERVEX_NOTEBOOK_CALL_LEDGER")
    return Path(env) if env else _DEFAULT_LEDGER_PATH


def _get_ledger():
    from visionservex.reporting.notebook_calls import NotebookCallLedger

    path = _resolve_ledger_path()
    if path.exists():
        return NotebookCallLedger.load(path)
    return NotebookCallLedger.init(path)


def record_model_call_simple(
    *,
    model_id: str,
    notebook: str,
    section: str,
    task: str,
    command: str,
    call_type: str = "smoke",
    status: str = "executed",
    final_state: str = "",
    blocker_code: str = "",
    evidence_artifact: str = "",
    family: str = "",
    extras: dict[str, Any] | None = None,
) -> None:
    """Thin wrapper that records one call into the ledger."""
    from visionservex.reporting.notebook_calls import record_model_call

    record_model_call(
        model_id=model_id,
        notebook=notebook,
        section=section,
        task=task,
        command=command,
        call_type=call_type,
        status=status,
        final_state=final_state,
        blocker_code=blocker_code,
        evidence_artifact=evidence_artifact,
        family=family,
        extras=extras,
        ledger=_get_ledger(),
    )


def record_skip_simple(
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
) -> None:
    """Record an allowed skip for a model that is intentionally not run."""
    from visionservex.reporting.notebook_calls import record_skip

    record_skip(
        model_id=model_id,
        notebook=notebook,
        section=section,
        task=task,
        reason=reason,
        family=family,
        blocker_code=blocker_code,
        final_state=final_state,
        evidence_artifact=evidence_artifact,
        ledger=_get_ledger(),
    )


@contextmanager
def track_model_call(
    *,
    model_id: str,
    notebook: str,
    section: str,
    task: str,
    call_type: str = "smoke",
    command: str = "",
    family: str = "",
    evidence_artifact: str = "",
) -> Iterator[dict[str, Any]]:
    """Context manager that records a model invocation with success/failure.

    Yields a mutable ``info`` dict the caller can populate. On exit, records
    one ledger row. If an exception bubbles up, records ``executed_blocked``
    plus the exception type.
    """
    info: dict[str, Any] = {
        "final_state": "",
        "blocker_code": "",
        "evidence_artifact": evidence_artifact,
        "extras": {},
    }
    try:
        yield info
        status = "executed"
        final_state = info.get("final_state") or "smoke_passed"
        record_model_call_simple(
            model_id=model_id,
            notebook=notebook,
            section=section,
            task=task,
            command=command,
            call_type=call_type,
            status=status,
            final_state=final_state,
            blocker_code=info.get("blocker_code", ""),
            evidence_artifact=info.get("evidence_artifact", ""),
            family=family,
            extras=info.get("extras", {}),
        )
    except Exception as exc:
        record_model_call_simple(
            model_id=model_id,
            notebook=notebook,
            section=section,
            task=task,
            command=command,
            call_type=call_type,
            status="executed_blocked",
            final_state=info.get("final_state") or "expected_blocker",
            blocker_code=info.get("blocker_code") or type(exc).__name__,
            evidence_artifact=info.get("evidence_artifact", ""),
            family=family,
            extras={"exception_type": type(exc).__name__, "exception_msg": str(exc)[:300]},
        )
        raise


__all__ = [
    "record_model_call_simple",
    "record_skip_simple",
    "track_model_call",
]
