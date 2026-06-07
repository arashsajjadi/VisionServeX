#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7 serial execution driver — runs every task in its own subprocess (CPU,
isolated, per-task timeout), writes incremental JSONL + a final ledger CSV.

ONE process, strictly serial => respects project resource-safety rules
(no concurrent heavy jobs, no background pytest).
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "notebook" / "99_final_report" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
RUNNER = ROOT / "scripts" / "v37_run_one.py"
JSONL = REPORTS / "v37_raw_results.jsonl"
LEDGER = REPORTS / "v37_new_model_execution_ledger.csv"

# per-task timeout (s); downloads/large CPU models get headroom
TIMEOUT = 420


def list_tasks() -> list[str]:
    out = subprocess.run([sys.executable, str(RUNNER), "--task", "LIST"],
                         capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    tasks = list_tasks()
    print(f"[v37_drive] {len(tasks)} tasks", flush=True)
    JSONL.write_text("")
    results = []
    tmpdir = str(Path.home() / ".cache" / "vsx_tmp")
    Path(tmpdir).mkdir(parents=True, exist_ok=True)
    env = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_DISABLE_TELEMETRY": "1",
           "TOKENIZERS_PARALLELISM": "false", "PATH": "/usr/bin:/bin:/usr/local/bin",
           "HOME": str(Path.home()), "TMPDIR": tmpdir}  # sda2 tmp — protect /tmp tmpfs quota
    import os
    full_env = {**os.environ, **env}
    for i, task in enumerate(tasks, 1):
        t0 = time.perf_counter()
        print(f"[{i}/{len(tasks)}] {task} ...", flush=True)
        try:
            p = subprocess.run([sys.executable, str(RUNNER), "--task", task],
                               capture_output=True, text=True, timeout=TIMEOUT, env=full_env)
            line = ""
            for ln in p.stdout.splitlines():
                if ln.startswith("V37_RESULT "):
                    line = ln[len("V37_RESULT "):]
            if line:
                rec = json.loads(line)
            else:
                rec = {"task": task, "status": "failed",
                       "error": "no V37_RESULT emitted",
                       "stderr_tail": p.stderr[-400:]}
        except subprocess.TimeoutExpired:
            rec = {"task": task, "status": "timeout",
                   "error": f"exceeded {TIMEOUT}s"}
        except Exception as e:
            rec = {"task": task, "status": "driver_error", "error": str(e)}
        rec["wall_s"] = round(time.perf_counter() - t0, 1)
        results.append(rec)
        with JSONL.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(f"    -> {rec.get('status')} ({rec.get('wall_s')}s) "
              f"{rec.get('error','')[:80]}", flush=True)

    # build ledger CSV
    cols = ["execution_id", "task", "model_id", "engine", "result_task", "status",
            "latency_ms", "metric", "artifact", "error"]
    with LEDGER.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for idx, r in enumerate(results, 1):
            mid = r.get("model_id") or r.get("hf_id") or r.get("pipeline_id") or \
                  r.get("variant") or r.get("task", "").split(":")[-1]
            metric = (r.get("mask_area") or r.get("embed_dim") or r.get("n_boxes")
                      or r.get("n_instances") or r.get("frames_tracked")
                      or r.get("depth_shape") or r.get("onnx_bytes") or "")
            w.writerow({
                "execution_id": idx, "task": r.get("task"), "model_id": mid,
                "engine": r.get("engine", ""), "result_task": r.get("task_name") or r.get("result_task") or r.get("task", ""),
                "status": r.get("status"), "latency_ms": r.get("latency_ms", ""),
                "metric": metric, "artifact": r.get("artifact", ""),
                "error": (r.get("error", "") or "")[:200],
            })
    ok = sum(1 for r in results if r.get("status") == "ok")
    blocked = sum(1 for r in results if r.get("status") == "blocked")
    failed = sum(1 for r in results if r.get("status") in ("failed", "timeout", "driver_error"))
    summary = {"total": len(results), "ok": ok, "blocked": blocked, "failed": failed}
    (REPORTS / "v37_execution_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[v37_drive] DONE {summary}", flush=True)


if __name__ == "__main__":
    main()
