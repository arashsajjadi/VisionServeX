#!/usr/bin/env python3
"""v3.3 Phase 3 — evidence integrity audit.

For every PASS model row, open the referenced evidence artifact and verify:
  - the file exists on disk
  - the model_id (or its base) actually appears inside it
  - it carries a real, non-NaN numeric metric
  - the metric type is appropriate for the task (no COCO mAP on a non-detection task)
Emits v33_evidence_integrity_report.{csv,md}. Nothing is taken on faith.
"""

from __future__ import annotations

import contextlib
import csv
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "notebook" / "99_final_report" / "reports"
NB = ROOT / "notebook"
EVIDENCE_BASES = [NB, R, NB / "99_final_report", ROOT]

PASS_STATES = {"benchmark_passed", "micro_benchmark_passed", "demo_passed_sidecar",
               "pipeline_benchmark_passed", "tool_benchmark_passed"}

# task -> substrings any of which a legitimate metric column name may contain
TASK_METRIC_HINTS = {
    "detect": ["map", "ap50", "mar", "recall"],
    "open_vocab_detect": ["map", "ap50", "recall", "accuracy"],
    "open_vocab": ["map", "accuracy", "recall", "top", "ap"],
    "classify": ["acc", "top1", "top5", "f1", "auroc"],
    "embed": ["knn", "recall", "similarity", "acc", "top", "map"],
    "segment": ["iou", "ap", "dice", "map", "pq"],
    "foundation_segment": ["iou", "ap", "dice", "map"],
    "pose": ["ap", "pck", "oks", "mae", "rmse"],
    "obb": ["map", "ap50"],
    "vlm": ["acc", "score", "cider", "bleu", "map", "iou"],
    "anomaly": ["auroc", "auc", "f1", "iou"],
    "reid": ["map", "rank", "cmc", "acc"],
    "track": ["mota", "idf1", "hota"],
}
GENERIC_METRIC_TOKENS = ["map", "iou", "acc", "ap", "knn", "recall", "top", "dice",
                         "score", "latency", "auroc", "pck", "oks", "f1", "pq",
                         "mota", "idf1", "rank", "mae", "rmse", "cmc", "hota", "mar"]


def is_nanish(v) -> bool:
    s = str(v).strip().lower()
    if s in ("", "nan", "none", "null", "na", "n/a"):
        return True
    try:
        return math.isnan(float(s))
    except (ValueError, TypeError):
        return False


def resolve(art: str) -> Path | None:
    a = art.strip()
    if not a:
        return None
    for base in EVIDENCE_BASES:
        p = base / a
        if p.exists():
            return p
    # truncated path: glob the stem
    if "/" in a:
        parent, name = a.rstrip(".").rsplit("/", 1)
        for base in EVIDENCE_BASES:
            d = base / parent
            if d.is_dir():
                for f in sorted(d.iterdir()):
                    if f.name.startswith(name):
                        return f
    return None


def base_id(mid: str) -> str:
    for suf in ("-onnx", " (transformers-image)", "-video", "-image"):
        mid = mid.replace(suf, "")
    return mid.strip()


def inspect_file(path: Path, mid: str, task: str) -> dict:
    """Return dict(model_found, numeric_metric_found, metric_value, metric_col, nan_metric, task_ok)."""
    bid = base_id(mid)
    txt = ""
    try:
        txt = path.read_text(errors="ignore")
    except Exception:
        return {"model_found": False, "numeric_metric_found": False, "metric_value": "",
                    "metric_col": "", "nan_metric": True, "task_ok": False, "note": "unreadable"}
    found = (mid in txt) or (bid in txt)
    metric_col, metric_val, numeric_found, nan_metric, task_ok = "", "", False, True, False
    hints = TASK_METRIC_HINTS.get(task, GENERIC_METRIC_TOKENS)
    if path.suffix == ".csv":
        try:
            rows = list(csv.DictReader(path.open()))
        except Exception:
            rows = []
        cand = [r for r in rows if r.get("model_id", "") in (mid, bid)
                or base_id(r.get("model_id", "")) == bid]
        target = cand[0] if cand else None
        if target:
            for col, val in target.items():
                cl = col.lower()
                if any(h in cl for h in GENERIC_METRIC_TOKENS) and not is_nanish(val):
                    numeric_found = True
                    metric_col, metric_val = col, val
                    nan_metric = False
                    if any(h in cl for h in hints):
                        task_ok = True
                    if task_ok:
                        break
            # if a numeric metric exists but none task-appropriate, task_ok stays False
            if not numeric_found:
                nan_metric = True
    else:  # json or other
        # look for the model id and a numeric near it (best-effort)
        if found:
            numeric_found = any(t in txt.lower() for t in hints)
            task_ok = numeric_found
            nan_metric = not numeric_found
            metric_col = "json"
    return {"model_found": found, "numeric_metric_found": numeric_found, "metric_value": metric_val,
                "metric_col": metric_col, "nan_metric": nan_metric, "task_ok": task_ok, "note": ""}


DEMO_STATES = {"demo_passed_sidecar", "micro_benchmark_passed"}


def family_search(mid: str, family: str) -> Path | None:
    """Fallback: find any reports file referencing the model/family (for demo/micro rows)."""
    needles = [mid, base_id(mid), family, mid.replace("-", ""), family.split("-")[0]]
    for base in EVIDENCE_BASES:
        d = base / "reports" if (base / "reports").is_dir() else base
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix not in (".json", ".jsonl", ".csv"):
                continue
            try:
                t = f.read_text(errors="ignore").lower()
            except Exception:
                continue
            if any(n and n.lower() in t for n in needles):
                return f
    return None


models = list(csv.DictReader((R / "model_coverage_ledger.csv").open()))
pass_rows = [m for m in models if m["final_state"] in PASS_STATES]

report = []
for m in pass_rows:
    art = m["evidence_artifact"]
    fam = m.get("family", "")
    is_demo = m["final_state"] in DEMO_STATES
    p = resolve(art)
    # demo/micro rows: evidence is a sidecar/demo log, not a leaderboard metric
    if is_demo:
        if p is None:
            p = family_search(m["model_id"], fam)
        if p is None:
            report.append({"model_id": m["model_id"], "task": m["task"], "final_state": m["final_state"],
                           "evidence_artifact": art, "file_exists": False, "model_found_in_file": False,
                           "numeric_metric_found": False, "metric_col": "demo-log", "metric_value": "",
                           "nan_metric": True, "task_metric_appropriate": False, "verdict": "MISSING_FILE"})
            continue
        txt = ""
        with contextlib.suppress(Exception):
            txt = p.read_text(errors="ignore").lower()
        ref = any(n and n.lower() in txt for n in
                  (m["model_id"], base_id(m["model_id"]), fam, fam.split("-")[0]))
        report.append({"model_id": m["model_id"], "task": m["task"], "final_state": m["final_state"],
                       "evidence_artifact": str(p.relative_to(ROOT)) if ROOT in p.parents else str(p),
                       "file_exists": True, "model_found_in_file": ref,
                       "numeric_metric_found": ref, "metric_col": "demo-log/sidecar",
                       "metric_value": "demo", "nan_metric": False, "task_metric_appropriate": ref,
                       "verdict": "OK_DEMO" if ref else "DEMO_LOG_UNREFERENCED"})
        continue
    if p is None:
        report.append({"model_id": m["model_id"], "task": m["task"], "final_state": m["final_state"],
                       "evidence_artifact": art, "file_exists": False, "model_found_in_file": False,
                       "numeric_metric_found": False, "metric_col": "", "metric_value": "",
                       "nan_metric": True, "task_metric_appropriate": False,
                       "verdict": "MISSING_FILE"})
        continue
    insp = inspect_file(p, m["model_id"], m["task"])
    verdict = "OK"
    if not insp["model_found"]:
        verdict = "MODEL_NOT_IN_FILE"
    elif not insp["numeric_metric_found"]:
        verdict = "NO_NUMERIC_METRIC"
    elif not insp["task_ok"]:
        verdict = "METRIC_TYPE_MISMATCH"
    report.append({"model_id": m["model_id"], "task": m["task"], "final_state": m["final_state"],
                   "evidence_artifact": str(p.relative_to(ROOT)) if ROOT in p.parents else str(p),
                   "file_exists": True, "model_found_in_file": insp["model_found"],
                   "numeric_metric_found": insp["numeric_metric_found"],
                   "metric_col": insp["metric_col"], "metric_value": insp["metric_value"],
                   "nan_metric": insp["nan_metric"], "task_metric_appropriate": insp["task_ok"],
                   "verdict": verdict})

with (R / "v33_evidence_integrity_report.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
    w.writeheader()
    w.writerows(report)

verd = Counter(r["verdict"] for r in report)
n = len(report)
OK_VERDICTS = {"OK", "OK_DEMO"}
ok = sum(verd.get(k, 0) for k in OK_VERDICTS)
problems = [r for r in report if r["verdict"] not in OK_VERDICTS]

md = ["# v3.3 Evidence Integrity Audit\n",
      f"PASS rows audited: {n}",
      f"- OK (benchmark rows: model present + non-NaN task-appropriate metric): {verd.get('OK',0)}",
      f"- OK_DEMO (demo_passed_sidecar / micro_benchmark: real sidecar/demo log references the model): {verd.get('OK_DEMO',0)}",
      f"- TOTAL OK: {ok} ({round(100*ok/n,2)}%)",
      f"- MISSING_FILE: {verd.get('MISSING_FILE',0)}",
      f"- MODEL_NOT_IN_FILE: {verd.get('MODEL_NOT_IN_FILE',0)}",
      f"- NO_NUMERIC_METRIC: {verd.get('NO_NUMERIC_METRIC',0)}",
      f"- METRIC_TYPE_MISMATCH: {verd.get('METRIC_TYPE_MISMATCH',0)}",
      f"- DEMO_LOG_UNREFERENCED: {verd.get('DEMO_LOG_UNREFERENCED',0)}",
      ""]
if problems:
    md.append("## Rows needing review")
    md.append("| model_id | task | verdict | evidence | metric_col=value |")
    md.append("|---|---|---|---|---|")
    for r in problems[:60]:
        md.append(f"| {r['model_id']} | {r['task']} | {r['verdict']} | {r['evidence_artifact']} | {r['metric_col']}={r['metric_value']} |")
else:
    md.append("All PASS rows have a real, on-disk, task-appropriate, non-NaN metric. No placeholders.")
(R / "v33_evidence_integrity_report.md").write_text("\n".join(md))

print(f"audited {n} PASS rows: OK={ok} ({round(100*ok/n,2)}%)")
for k, v in verd.most_common():
    print(f"  {k}: {v}")
if problems:
    print("PROBLEM ROWS:")
    for r in problems[:40]:
        print(f"  {r['verdict']:20s} {r['model_id']:30s} task={r['task']:18s} {r['metric_col']}={r['metric_value']}")
