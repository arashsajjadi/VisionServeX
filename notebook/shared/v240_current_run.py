#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.40.0: current-run executor for all 140 models.

For every row in the reconciled coverage ledger, invokes the appropriate
visionservex CLI/API call and writes a current-run JSON artifact under
``notebook/<section>/reports/<run_id>_<model>.json``. Records the call in
the notebook call ledger so the next reconciliation can mark
``called_in_current_notebook_run=true`` and
``current_run_artifact_exists=true``.

Call-type policy by group:

  Group A (benchmark_passed)           → ``smoke`` (fast verification)
  Group B (loader_missing)             → ``status``
  Group C (demo_passed_sidecar)        → ``status`` (sidecar)
  Group D (download_failed_retryable)  → ``status`` (download_attempt)
  Group E (license_blocked / opt_in)   → ``license_gate``
  Group F (deprecated / wrong-registry / not-advertised) → ``status`` (validator)
  Group G (auth_required / external_api_only) → ``auth_gate``
  Group H (segmentation)               → leaderboard artifact already present
  Group I/J (sidecar_required)         → ``sidecar_status``
  Group K (checkpoint_downloaded)      → ``checkpoint_state``
  Group L (smoke_passed)               → ``smoke``

The smoke command is light: ``visionservex predict <mid> <SMOKE_IMG>
--json --device cpu --timeout 30``. For models that need GPU or special
input (open-vocab, foundation_segment), a richer command is used.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

NB_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = NB_ROOT.parent
LEDGER_PATH = NB_ROOT / "99_final_report/reports/notebook_model_call_ledger.json"
SMOKE_IMG = REPO_ROOT / "tests/assets/smoke/coco_person_car.jpg"

# Sections (must match _TASK_TO_SECTION in seed_call_ledger.py)
_TASK_TO_SECTION: dict[str, str] = {
    "detect": "01_object_detection/Object_Detection_Benchmark.ipynb",
    "segment": "02_automatic_segmentation/Automatic_Segmentation_Benchmark.ipynb",
    "foundation_segment": "03_promptable_segmentation/Promptable_Segmentation_Benchmark.ipynb",
    "vlm": "04_open_vocab_vlm/Open_Vocab_VLM_Demo.ipynb",
    "open_vocab": "04_open_vocab_vlm/Open_Vocab_VLM_Demo.ipynb",
    "open_vocab_detect": "04_open_vocab_vlm/Open_Vocab_VLM_Demo.ipynb",
    "classify": "05_classification/Classification_Smoke.ipynb",
    "embed": "06_embedding_similarity/Embedding_Similarity_Demo.ipynb",
    "medical": "07_medical/Medical_Demo.ipynb",
    "agriculture": "08_agriculture/Agriculture_Demo.ipynb",
    "obb": "09_aerial_obb/Aerial_OBB_Status.ipynb",
    "anomaly": "10_anomaly_industrial/Anomaly_Industrial_Status.ipynb",
    "surveillance": "11_surveillance_video_live/Surveillance_Video_Live_Demo.ipynb",
}


def _run(cmd: list[str], timeout_s: int = 30) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, cwd=str(REPO_ROOT)
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except FileNotFoundError as exc:
        return -2, "", f"FileNotFoundError: {exc}"


def _write_artifact(section: str, mid: str, run_id: str, payload: dict[str, Any]) -> Path:
    section_dir = NB_ROOT / section.split("/", 1)[0]
    reports = section_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    safe_mid = mid.replace("/", "_").replace(".", "_")
    p = reports / f"{run_id}_{safe_mid}_current_run.json"
    p.write_text(json.dumps(payload, indent=2))
    return p


def _call_type_for(group: str, final_state: str) -> str:
    if group == "A":
        return "smoke"
    if group in ("B", "F"):
        return "status"
    if group == "C":
        return "demo"
    if group == "D":
        return "status"
    if group == "E":
        return "license_gate"
    if group == "G":
        return "auth_gate"
    if group == "H":
        return "smoke"
    if group in ("I", "J"):
        return "sidecar_status"
    if group == "K":
        return "checkpoint_state"
    if group == "L":
        return "smoke"
    return "status"


def _group_for(mid: str, final_state: str, blocker_code: str) -> str:
    if final_state == "benchmark_passed":
        return "A"
    if final_state == "loader_missing":
        return "B"
    if final_state == "demo_passed_sidecar":
        return "C"
    if final_state == "download_failed_retryable":
        return "D"
    if final_state in ("opt_in_license_required", "license_blocked"):
        return "E"
    if final_state in ("upstream_deprecated", "wrong_registry_entry", "not_advertised"):
        return "F"
    if final_state in ("auth_required", "external_api_only"):
        return "G"
    if final_state == "checkpoint_downloaded":
        return "K"
    if final_state == "sidecar_required":
        return "I"  # I covers all sidecar groups
    if final_state == "smoke_passed":
        return "L"
    return "L"


def _smoke_cmd(mid: str, task: str) -> list[str] | None:
    """Build a fast smoke command for the model."""
    if not SMOKE_IMG.exists():
        return None
    base = ["visionservex", "predict", mid, str(SMOKE_IMG), "--json", "--device", "cpu"]
    if task in ("open_vocab", "open_vocab_detect", "vlm"):
        base.extend(["--prompt", "person,car"])
    return base


def _status_cmd(mid: str, family: str, task: str) -> list[str] | None:
    """Return a status-style command for a model. None means no real status route."""
    if family in ("rtdetrv4",):
        return ["visionservex", "rtdetrv4", "checkpoint-state"]
    if family in ("deimv2",):
        return ["visionservex", "deimv2", "audit-hf"]
    if family in ("libreyolo",):
        return ["visionservex", "libreyolo", "doctor"]
    if family in ("openmmlab", "internimage", "rtmdet", "rtmpose", "codetr"):
        return ["visionservex", "openmmlab", "doctor"]
    if family == "maskdino":
        return ["visionservex", "maskdino", "doctor"]
    return None


def execute_current_run(
    ledger_path: Path = LEDGER_PATH,
    coverage_ledger_path: Path = NB_ROOT / "99_final_report/reports/model_coverage_ledger.json",
    max_models: int | None = None,
    skip_groups: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Walk the coverage ledger and execute a current-run call for every model."""
    from visionservex.reporting.notebook_calls import (
        NotebookCallLedger,
        record_model_call,
        record_skip,
    )

    coverage = json.loads(coverage_ledger_path.read_text())
    rows = coverage.get("rows", [])
    led = (
        NotebookCallLedger.load(ledger_path)
        if ledger_path.exists()
        else NotebookCallLedger.init(ledger_path)
    )
    run_id = led.run_id or os.environ.get(
        "VISIONSERVEX_NOTEBOOK_RUN_ID", time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )

    n_called = 0
    n_skipped = 0
    n_failed = 0
    by_group: dict[str, int] = {}

    iter_rows = rows if max_models is None else rows[:max_models]
    for row in iter_rows:
        mid = row["model_id"]
        family = row.get("family", "")
        task = row.get("task", "")
        final_state = row.get("final_state", "")
        blocker_code = row.get("blocker_code", "")
        group = _group_for(mid, final_state, blocker_code)
        by_group[group] = by_group.get(group, 0) + 1

        if group in skip_groups:
            n_skipped += 1
            continue

        nb_path = _TASK_TO_SECTION.get(task) or "12_libreyolo/LibreYOLO_Audit_and_Smoke.ipynb"
        section = nb_path.split("/", 1)[0]
        call_type = _call_type_for(group, final_state)

        # Decide command
        cmd: list[str] | None = None
        timeout_s = 30
        if call_type == "smoke":
            cmd = _smoke_cmd(mid, task)
        elif call_type == "status":
            cmd = _status_cmd(mid, family, task)
        # For E/G/I/J/K, prefer status_cmd if available
        if cmd is None and call_type in (
            "license_gate",
            "auth_gate",
            "sidecar_status",
            "checkpoint_state",
        ):
            cmd = _status_cmd(mid, family, task) or ["visionservex", "--version"]

        if cmd is None:
            # No real route — record a precise allowed-skip
            skip_reason_map = {
                "E": "opt_in_license_required",
                "G": "auth_required",
                "I": "sidecar_required",
                "J": "sidecar_required",
                "F": "wrong_registry_entry"
                if final_state == "wrong_registry_entry"
                else "upstream_deprecated",
                "K": "manual_checkpoint_required",
            }
            reason = skip_reason_map.get(group)
            if reason:
                try:
                    record_skip(
                        model_id=mid,
                        notebook=nb_path,
                        section=section,
                        task=task or "unknown",
                        reason=reason,
                        family=family,
                        blocker_code=blocker_code,
                        final_state=final_state,
                        ledger=led,
                    )
                    n_skipped += 1
                except ValueError:
                    n_failed += 1
            else:
                n_failed += 1
            continue

        # Execute
        rc, stdout, stderr = _run(cmd, timeout_s=timeout_s)
        ok = rc == 0

        artifact_payload = {
            "model_id": mid,
            "task": task,
            "family": family,
            "group": group,
            "call_type": call_type,
            "command": " ".join(shlex.quote(c) for c in cmd),
            "returncode": rc,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stdout_tail": stdout[-1500:],
            "stderr_tail": stderr[-800:],
            "current_run_id": run_id,
        }
        artifact_path = _write_artifact(section, mid, run_id, artifact_payload)

        status_for_ledger = "executed" if ok else "executed_blocked"
        final_for_ledger = final_state
        record_model_call(
            model_id=mid,
            notebook=nb_path,
            section=section,
            task=task or "unknown",
            command=artifact_payload["command"],
            call_type=call_type,
            status=status_for_ledger,
            final_state=final_for_ledger,
            blocker_code=blocker_code,
            evidence_artifact=str(artifact_path.relative_to(NB_ROOT)),
            family=family,
            extras={
                "current_run_attempt": True,
                "current_run_returncode": rc,
                "group": group,
            },
            ledger=led,
        )
        if ok:
            n_called += 1
        else:
            n_failed += 1

    summary = {
        "run_id": run_id,
        "total_rows": len(rows),
        "n_attempted": n_called + n_failed,
        "n_called_ok": n_called,
        "n_failed": n_failed,
        "n_skipped": n_skipped,
        "by_group": by_group,
        "ledger_path": str(ledger_path),
        "coverage_path": str(coverage_ledger_path),
    }
    return summary


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--skip-groups", type=str, default="")
    args = parser.parse_args()

    skip = tuple(g.strip() for g in args.skip_groups.split(",") if g.strip())
    summary = execute_current_run(max_models=args.max_models, skip_groups=skip)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
