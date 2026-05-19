# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.39.0: shared wrapper for executing visionservex CLI calls from notebooks.

All notebooks should go through ``run_vsx(...)`` instead of calling
``subprocess.run(...)`` directly. It prints the command, captures output,
parses JSON, writes a per-notebook command log, and records one row into
the notebook call ledger.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from notebook_call_tracker import record_model_call_simple, record_skip_simple

_DEFAULT_NB_ROOT = Path(__file__).resolve().parent.parent


def _commands_log_dir(notebook: str) -> Path:
    nb_path = Path(notebook)
    if not nb_path.is_absolute():
        nb_path = _DEFAULT_NB_ROOT / nb_path
    return nb_path.parent / "commands"


def _write_command_log(notebook: str, model_id: str, payload: dict[str, Any]) -> Path:
    d = _commands_log_dir(notebook)
    d.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    out = d / f"{stamp}_{model_id}.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def _safe_loads(text: str) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def run_vsx(
    *,
    model_id: str,
    task: str,
    call_type: str,
    cmd: list[str],
    notebook: str,
    section: str,
    expected_artifacts: list[str] | None = None,
    family: str = "",
    timeout_s: int = 1200,
    cwd: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    final_state_on_success: str = "",
) -> dict[str, Any]:
    """Run a VisionServeX CLI command, record the call, return a structured result.

    The function never raises: failures are surfaced as ``status='failed'``
    and recorded as ``execution_status='executed_blocked'`` in the ledger.
    """
    started = time.time()
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    print(f"[run_vsx] {model_id} :: {cmd_str}")
    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
        returncode = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        returncode = -1
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (
            exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        ) + "\nTIMEOUT_EXPIRED"
    except FileNotFoundError as exc:
        returncode = -2
        stdout = ""
        stderr = f"FileNotFoundError: {exc}"

    duration = round(time.time() - started, 2)
    parsed = _safe_loads(stdout)
    parsed_status = ""
    parsed_code = ""
    if isinstance(parsed, dict):
        parsed_status = str(parsed.get("status", ""))
        parsed_code = str(parsed.get("code", "") or parsed.get("blocker_code", ""))

    status = "ok" if returncode == 0 else "failed"
    final_state = final_state_on_success or ""
    blocker_code = ""
    if status == "ok":
        if not final_state:
            mapping = {
                "benchmark": "benchmark_passed",
                "contract": "contract_passed",
                "smoke": "smoke_passed",
                "demo": "demo_passed_sidecar",
                "doctor": "smoke_ok_no_metric",
                "status": "smoke_ok_no_metric",
                "checkpoint_state": "checkpoint_downloaded",
                "auth_gate": "smoke_ok_no_metric",
                "sidecar_status": "smoke_ok_no_metric",
            }
            final_state = mapping.get(call_type, "smoke_passed")
        if parsed_status and parsed_status not in {"ok", "success"}:
            final_state = parsed_status
            blocker_code = parsed_code
    else:
        final_state = parsed_status or "expected_blocker"
        blocker_code = parsed_code or "SUBPROCESS_NONZERO"
        if returncode == -1:
            blocker_code = "PROCESS_TIMEOUT"

    # Resolve evidence artifact
    evidence_artifact = ""
    if expected_artifacts:
        for art in expected_artifacts:
            if Path(art).exists():
                evidence_artifact = art
                break
        if not evidence_artifact:
            evidence_artifact = expected_artifacts[0]

    log_path = _write_command_log(
        notebook,
        model_id,
        {
            "model_id": model_id,
            "task": task,
            "call_type": call_type,
            "notebook": notebook,
            "section": section,
            "command": cmd_str,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "duration_sec": duration,
            "returncode": returncode,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
            "parsed_stdout": parsed if isinstance(parsed, (dict, list)) else None,
            "final_state": final_state,
            "blocker_code": blocker_code,
            "evidence_artifact": evidence_artifact,
            "expected_artifacts": expected_artifacts or [],
        },
    )

    record_model_call_simple(
        model_id=model_id,
        notebook=notebook,
        section=section,
        task=task,
        command=cmd_str,
        call_type=call_type,
        status="executed" if status == "ok" else "executed_blocked",
        final_state=final_state,
        blocker_code=blocker_code,
        evidence_artifact=evidence_artifact or str(log_path),
        family=family,
        extras={"duration_sec": duration, "returncode": returncode},
    )

    return {
        "status": status,
        "model_id": model_id,
        "final_state": final_state,
        "blocker_code": blocker_code,
        "returncode": returncode,
        "duration_sec": duration,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "parsed_stdout": parsed,
        "evidence_artifact": evidence_artifact,
        "command_log": str(log_path),
    }


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
) -> None:
    """Pass-through to record an allowed skip for a model not run in this notebook."""
    record_skip_simple(
        model_id=model_id,
        notebook=notebook,
        section=section,
        task=task,
        reason=reason,
        family=family,
        blocker_code=blocker_code,
        final_state=final_state,
        evidence_artifact=evidence_artifact,
    )


__all__ = ["record_skip", "run_vsx"]
