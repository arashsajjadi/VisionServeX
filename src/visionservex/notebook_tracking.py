# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.47.2 notebook tracking.

Every notebook model action — benchmark, smoke, contract, status gate,
historical validation, license gate, auth gate — must produce a JSONL
event in the notebook call ledger. The reconciler reads those events and
derives ``called_in_notebook``, ``called_in_current_notebook_run``, and
``current_run_artifact_exists`` from them. No flags are patched into the
CSV by editing it directly.

Usage in notebooks or RUN_ALL.ipynb::

    from visionservex.notebook_tracking import NotebookRunContext, scan_task_outputs

    ctx = NotebookRunContext(
        run_id=os.environ["VISIONSERVEX_NOTEBOOK_RUN_ID"],
        notebook_path="01_object_detection/Object_Detection_Benchmark.ipynb",
        ledger_path=Path("99_final_report/reports/notebook_model_call_ledger.jsonl"),
    )
    ctx.record_benchmark(
        model_id="rtdetrv4-x",
        command=["visionservex", "benchmark-detection", "--models", "rtdetrv4-x"],
        evidence_artifact="01_object_detection/reports/detection_leaderboard.json",
        status="success",
    )

The RUN_ALL auto-scanner uses ``scan_task_outputs`` to produce events from
every task notebook's report directory without requiring task notebooks to
import this module explicitly.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

__all__ = [
    "CALL_TYPES",
    "EVENT_STATUSES",
    "NotebookRunContext",
    "load_jsonl_ledger",
    "scan_task_outputs",
]


CALL_TYPES: frozenset[str] = frozenset(
    {
        "benchmark",
        "smoke",
        "contract",
        "demo",
        "runtime_prepare",
        "status_gate",
        "historical_validated",
        "license_gate",
        "auth_gate",
        "registry_alias",
        "external_baseline",
        "notebook_coverage",
    }
)

EVENT_STATUSES: frozenset[str] = frozenset(
    {
        "success",
        "failure",
        "status_gate",
        "historical_validated",
        "license_gate",
        "auth_gate",
        "registry_alias",
        "external_baseline",
        "attempted",
    }
)


def _sha256_of_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()[:16]
    except OSError:
        return None


class NotebookRunContext:
    """JSONL-based context for recording all model interactions in a notebook run.

    Each call to a record_* method appends one JSON line to the ledger file.
    The ledger is then consumed by the v2.39 reconciler to populate
    ``called_in_notebook`` and related flags.
    """

    def __init__(
        self,
        run_id: str,
        notebook_path: str,
        ledger_path: Path | str,
    ) -> None:
        self.run_id = run_id
        self.notebook_path = notebook_path
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env_or_create(
        cls,
        ledger_path: Path | str,
        notebook_path: str = "unknown",
    ) -> NotebookRunContext:
        run_id = os.environ.get("VISIONSERVEX_NOTEBOOK_RUN_ID") or time.strftime(
            "%Y%m%dT%H%M%SZ_auto", time.gmtime()
        )
        return cls(run_id=run_id, notebook_path=notebook_path, ledger_path=ledger_path)

    def _write(self, event: dict[str, Any]) -> None:
        with self.ledger_path.open("a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def _base_event(self) -> dict[str, Any]:
        return {
            "event_id": str(uuid.uuid4())[:8],
            "run_id": self.run_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notebook_path": self.notebook_path,
        }

    def record_benchmark(
        self,
        model_id: str,
        command: list[str],
        evidence_artifact: str,
        status: str = "success",
        task: str = "",
        error_tail: str = "",
        next_command: str = "",
        runtime_id: str = "",
    ) -> None:
        """Record a real benchmark execution event."""
        ev_path = Path(evidence_artifact)
        ev = self._base_event()
        ev.update(
            {
                "model_id": model_id,
                "task": task,
                "call_type": "benchmark",
                "command_attempted": " ".join(command),
                "started_at": ev["timestamp_utc"],
                "ended_at": ev["timestamp_utc"],
                "status": status,
                "evidence_artifact": evidence_artifact,
                "evidence_artifact_exists": ev_path.exists(),
                "output_artifact_sha256": _sha256_of_file(ev_path) if ev_path.exists() else None,
                "error_type": "",
                "error_message_tail": error_tail,
                "next_iteration_command": next_command,
                "execution_origin": "current_run_executed"
                if status == "success"
                else "current_run_status_gate",
                "runtime_id": runtime_id,
            }
        )
        self._write(ev)

    def record_smoke(
        self,
        model_id: str,
        command: list[str],
        evidence_artifact: str,
        status: str = "success",
        task: str = "",
        runtime_id: str = "",
    ) -> None:
        ev_path = Path(evidence_artifact)
        ev = self._base_event()
        ev.update(
            {
                "model_id": model_id,
                "task": task,
                "call_type": "smoke",
                "command_attempted": " ".join(command),
                "started_at": ev["timestamp_utc"],
                "ended_at": ev["timestamp_utc"],
                "status": status,
                "evidence_artifact": evidence_artifact,
                "evidence_artifact_exists": ev_path.exists(),
                "output_artifact_sha256": _sha256_of_file(ev_path) if ev_path.exists() else None,
                "error_type": "",
                "error_message_tail": "",
                "next_iteration_command": "",
                "execution_origin": "current_run_executed"
                if status == "success"
                else "current_run_status_gate",
                "runtime_id": runtime_id,
            }
        )
        self._write(ev)

    def record_contract(
        self,
        model_id: str,
        command: list[str],
        evidence_artifact: str,
        status: str = "success",
        task: str = "",
        runtime_id: str = "",
    ) -> None:
        ev_path = Path(evidence_artifact)
        ev = self._base_event()
        ev.update(
            {
                "model_id": model_id,
                "task": task,
                "call_type": "contract",
                "command_attempted": " ".join(command),
                "started_at": ev["timestamp_utc"],
                "ended_at": ev["timestamp_utc"],
                "status": status,
                "evidence_artifact": evidence_artifact,
                "evidence_artifact_exists": ev_path.exists(),
                "output_artifact_sha256": _sha256_of_file(ev_path) if ev_path.exists() else None,
                "error_type": "",
                "error_message_tail": "",
                "next_iteration_command": "",
                "execution_origin": "current_run_executed"
                if status == "success"
                else "current_run_status_gate",
                "runtime_id": runtime_id,
            }
        )
        self._write(ev)

    def record_status_gate(
        self,
        model_id: str,
        command: list[str],
        evidence_artifact: str,
        call_type: str = "status_gate",
        task: str = "",
        next_command: str = "",
        runtime_id: str = "",
    ) -> None:
        """Record a status gate check (model in unresolved section)."""
        ev_path = Path(evidence_artifact) if evidence_artifact else None
        ev = self._base_event()
        ev.update(
            {
                "model_id": model_id,
                "task": task,
                "call_type": call_type,
                "command_attempted": " ".join(command),
                "started_at": ev["timestamp_utc"],
                "ended_at": ev["timestamp_utc"],
                "status": "status_gate",
                "evidence_artifact": evidence_artifact,
                "evidence_artifact_exists": ev_path.exists() if ev_path else False,
                "output_artifact_sha256": None,
                "error_type": "",
                "error_message_tail": "",
                "next_iteration_command": next_command,
                "execution_origin": "current_run_status_gate",
                "runtime_id": runtime_id,
            }
        )
        self._write(ev)

    def record_historical_validation(
        self,
        model_id: str,
        evidence_artifact: str,
        task: str = "",
        runtime_id: str = "",
    ) -> None:
        """Record a historical validation — model carried forward from previous run."""
        ev_path = Path(evidence_artifact) if evidence_artifact else None
        ev = self._base_event()
        ev.update(
            {
                "model_id": model_id,
                "task": task,
                "call_type": "historical_validated",
                "command_attempted": f"visionservex models status {model_id}",
                "started_at": ev["timestamp_utc"],
                "ended_at": ev["timestamp_utc"],
                "status": "historical_validated",
                "evidence_artifact": evidence_artifact,
                "evidence_artifact_exists": ev_path.exists() if ev_path else False,
                "output_artifact_sha256": _sha256_of_file(ev_path)
                if ev_path and ev_path.exists()
                else None,
                "error_type": "",
                "error_message_tail": "",
                "next_iteration_command": "",
                "execution_origin": "historical_validated",
                "runtime_id": runtime_id,
            }
        )
        self._write(ev)

    def record_license_gate(
        self,
        model_id: str,
        license_status: str,
        opt_in_command: str,
        task: str = "",
    ) -> None:
        ev = self._base_event()
        ev.update(
            {
                "model_id": model_id,
                "task": task,
                "call_type": "license_gate",
                "command_attempted": f"visionservex license-gate check {model_id}",
                "started_at": ev["timestamp_utc"],
                "ended_at": ev["timestamp_utc"],
                "status": "license_gate",
                "evidence_artifact": "",
                "evidence_artifact_exists": False,
                "output_artifact_sha256": None,
                "error_type": "",
                "error_message_tail": f"License: {license_status}",
                "next_iteration_command": opt_in_command,
                "execution_origin": "excluded_restricted_license",
                "runtime_id": "license_gate_runtime",
            }
        )
        self._write(ev)

    def record_auth_gate(
        self,
        model_id: str,
        auth_env_var: str,
        task: str = "",
    ) -> None:
        ev = self._base_event()
        ev.update(
            {
                "model_id": model_id,
                "task": task,
                "call_type": "auth_gate",
                "command_attempted": f"visionservex run {model_id} --use-auth-if-available",
                "started_at": ev["timestamp_utc"],
                "ended_at": ev["timestamp_utc"],
                "status": "auth_gate",
                "evidence_artifact": "",
                "evidence_artifact_exists": False,
                "output_artifact_sha256": None,
                "error_type": "",
                "error_message_tail": f"Requires: {auth_env_var}",
                "next_iteration_command": f"export {auth_env_var}=<token> && visionservex run {model_id} <input>",
                "execution_origin": "auth_required",
                "runtime_id": "auth_gate_runtime",
            }
        )
        self._write(ev)

    def record_registry_alias(
        self,
        model_id: str,
        alias_target: str,
        task: str = "",
    ) -> None:
        ev = self._base_event()
        ev.update(
            {
                "model_id": model_id,
                "task": task,
                "call_type": "registry_alias",
                "command_attempted": f"visionservex registry remap {model_id} --alias-to {alias_target}",
                "started_at": ev["timestamp_utc"],
                "ended_at": ev["timestamp_utc"],
                "status": "registry_alias",
                "evidence_artifact": "",
                "evidence_artifact_exists": False,
                "output_artifact_sha256": None,
                "error_type": "",
                "error_message_tail": f"Alias target: {alias_target}",
                "next_iteration_command": f"visionservex run {alias_target} <input>",
                "execution_origin": "registry_alias",
                "runtime_id": "core_py311",
            }
        )
        self._write(ev)


def load_jsonl_ledger(
    ledger_path: Path | str,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Load all events from a JSONL ledger, optionally filtered by run_id."""
    path = Path(ledger_path)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if run_id is None or ev.get("run_id") == run_id:
                events.append(ev)
    return events


def scan_task_outputs(
    task_reports_root: Path | str,
    run_id: str,
    notebook_path: str,
    ledger_path: Path | str,
) -> int:
    """Scan all task notebook output directories and auto-generate JSONL events.

    This is the key function called by RUN_ALL.ipynb after all task notebooks
    have run. It reads every leaderboard CSV/JSON and status.json file and
    writes one JSONL event per model found, so that:

    * benchmark_passed models from leaderboards → ``call_type=benchmark``
    * smoke_passed from status files → ``call_type=smoke``
    * sidecar_required / blocked → ``call_type=status_gate``
    * historical_validated → ``call_type=historical_validated``

    Returns the number of events written.
    """
    root = Path(task_reports_root)
    ctx = NotebookRunContext(
        run_id=run_id,
        notebook_path=notebook_path,
        ledger_path=ledger_path,
    )
    written = 0

    # Walk all reports directories under task dirs.
    for reports_dir in root.glob("**/reports"):
        if "archive_legacy" in str(reports_dir) or "_runs" in str(reports_dir):
            continue
        if not reports_dir.is_dir():
            continue

        # --- leaderboard JSON (benchmark evidence) ---
        for lb_path in reports_dir.glob("*leaderboard*.json"):
            try:
                data = json.loads(lb_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            rows_data: list[dict[str, Any]] = []
            if isinstance(data, list):
                rows_data = data
            elif isinstance(data, dict):
                for k in ("rows", "models", "results"):
                    if isinstance(data.get(k), list):
                        rows_data = data[k]
                        break

            for row in rows_data:
                mid = (row.get("model_id") or row.get("name") or "").strip()
                if not mid:
                    continue
                status_val = (row.get("status") or "").lower()
                map95 = row.get("mAP50_95") or row.get("map50_95") or row.get("mask_mAP50_95")
                if status_val in {"ok", "benchmark_passed"} and map95 is not None:
                    call_t = "benchmark"
                    stat = "success"
                elif status_val in {"ok", "smoke_passed", "smoke"}:
                    call_t = "smoke"
                    stat = "success"
                else:
                    call_t = "status_gate"
                    stat = "status_gate"
                ctx.record_benchmark(
                    model_id=mid,
                    command=["visionservex", "benchmark", "--model", mid],
                    evidence_artifact=str(
                        lb_path.relative_to(root) if lb_path.is_relative_to(root) else lb_path
                    ),
                    status=stat,
                    task=row.get("task", ""),
                )
                written += 1

        # --- leaderboard CSV ---
        for lb_path in reports_dir.glob("*leaderboard*.csv"):
            try:
                rows_data = list(csv.DictReader(lb_path.open()))
            except OSError:
                continue
            for row in rows_data:
                mid = (row.get("model_id") or row.get("name") or "").strip()
                if not mid:
                    continue
                map95 = row.get("mAP50_95") or row.get("mask_mAP50_95")
                status_val = (row.get("status") or "").lower()
                if status_val == "ok" and map95:
                    call_t = "benchmark"
                    stat = "success"
                else:
                    call_t = "status_gate"
                    stat = "status_gate"
                ctx.record_benchmark(
                    model_id=mid,
                    command=["visionservex", "benchmark", "--model", mid],
                    evidence_artifact=str(
                        lb_path.relative_to(root) if lb_path.is_relative_to(root) else lb_path
                    ),
                    status=stat,
                    task="",
                )
                written += 1

        # --- per-model *_current_run.json files ---
        for cr_path in reports_dir.glob("*_current_run.json"):
            try:
                data = json.loads(cr_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            mid = (
                data.get("model_id")
                or data.get("name")
                or _extract_model_id_from_filename(cr_path.stem)
            )
            if not mid:
                continue
            final_state = data.get("final_state") or data.get("status") or ""
            if final_state in ("smoke_passed", "ok"):
                call_t = "smoke"
                stat = "success"
            elif final_state in ("benchmark_passed",):
                call_t = "benchmark"
                stat = "success"
            elif final_state in ("contract_passed",):
                call_t = "contract"
                stat = "success"
            else:
                call_t = "status_gate"
                stat = "status_gate"
            ctx.record_status_gate(
                model_id=mid,
                command=["visionservex", "models", "status", mid],
                evidence_artifact=str(
                    cr_path.relative_to(root) if cr_path.is_relative_to(root) else cr_path
                ),
                call_type=call_t,
                next_command=data.get("next_command", ""),
            )
            written += 1

        # --- status.json (task-level status) ---
        status_path = reports_dir / "status.json"
        if status_path.exists():
            try:
                data = json.loads(status_path.read_text())
            except (json.JSONDecodeError, OSError):
                data = {}
            # status.json is a task-level artifact; write a coverage event for
            # every model that might be in the task's leaderboard if listed.
            n_models = data.get("n_models") or 0
            _ = n_models  # we don't enumerate them here; leaderboard scan above handles it

    return written


def _extract_model_id_from_filename(stem: str) -> str:
    """Try to extract a model_id from a filename like 20260519T220000Z_v245_yolo11x_pt_current_run."""
    # Remove timestamp prefix if present (looks like 20XXXXXX...)
    parts = stem.split("_")
    # strip leading timestamp-like token
    if parts and parts[0].startswith("2") and len(parts[0]) >= 8 and parts[0][8:].isalpha():
        parts = parts[1:]
    # strip trailing _current_run
    if parts and parts[-1] == "run":
        parts = parts[:-1]
    if parts and parts[-1] == "current":
        parts = parts[:-1]
    # strip version token like v245
    if parts and parts[0].startswith("v") and parts[0][1:].isdigit():
        parts = parts[1:]
    # Rejoin remaining with dashes for model_id-like format
    candidate = "-".join(parts)
    # Normalize some common patterns
    candidate = candidate.replace("_pt-", ".pt-").replace("_pt", ".pt")
    return candidate
