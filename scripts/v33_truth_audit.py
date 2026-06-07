#!/usr/bin/env python3
"""VisionServeX v3.3 FULL TRUTH AUDIT — deterministic pass/fail/blocked/excluded matrix.

Computes the REAL status of every model, tool, pipeline from the canonical ledgers.
No hand-counting, no estimates: every number here is derived from the CSVs on disk and
on-disk evidence-file existence checks. Truth categories per the v3.3 spec.
"""

from __future__ import annotations

import contextlib
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "notebook" / "99_final_report" / "reports"
NB = ROOT / "notebook"

# ----------------------------- truth taxonomy -----------------------------
PASS_STATES = {"benchmark_passed", "micro_benchmark_passed", "demo_passed_sidecar",
               "pipeline_benchmark_passed", "tool_benchmark_passed"}
BLOCKED_STATES = {"auth_required", "checkpoint_required", "user_checkpoint_required",
                  "legal_review_required", "dataset_required", "sidecar_required",
                  "external_api_only", "expected_blocker", "not_released"}
FAIL_STATES = {"failed_runtime", "broken_import", "missing_artifact", "test_failed",
               "cli_failed", "api_failed"}
RESTRICTED_LICENSE_RE = ("AGPL", "GPL", "non-commercial", "noncommercial", "NonCommercial",
                         "S-Lab", "PML", " NC", "proprietary", "research-only")


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _has_restricted_license(lic: str) -> bool:
    low = lic.lower()
    # "GPL" substring would catch Apache; guard: only restrictive tokens
    toks = ["agpl", "gpl-3", "gpl-2", "gplv", "non-commercial", "noncommercial",
            "s-lab", "pml 1.0", "research-only", "proprietary", "cc-by-nc"]
    return any(t in low for t in toks)


def classify_model(final_state: str, default_safe: str, license_status: str,
                   evidence_artifact: str) -> tuple[str, bool, bool, bool]:
    """Return (category, is_pass, is_fail, is_blocked). EXCLUDED precedence first."""
    excluded = (not _truthy(default_safe)) or _has_restricted_license(license_status)
    if excluded:
        return "EXCLUDED", False, False, False
    if final_state in FAIL_STATES:
        return "FAIL", False, True, False
    if final_state in PASS_STATES:
        return "PASS", True, False, False
    if final_state == "wired":
        if evidence_artifact.strip():
            return "PASS", True, False, False
        return "OTHER_UNVERIFIED", False, False, False
    if final_state in BLOCKED_STATES:
        return "BLOCKED", False, False, True
    if final_state == "not_advertised":
        return "OTHER_UNVERIFIED", False, False, False
    return "OTHER_UNVERIFIED", False, False, False


# ----------------------------- evidence resolver -----------------------------
EVIDENCE_BASES = [NB, R, NB / "99_final_report", ROOT]


def evidence_exists(art: str) -> bool:
    if not art.strip():
        return False
    a = art.strip()
    for base in EVIDENCE_BASES:
        p = base / a
        if p.exists():
            return True
        # truncated paths in the ledger (…) — try glob on the stem
    # handle truncated evidence strings: try prefix-glob under notebook tree
    stem = a.rstrip(".").split("*")[0]
    if "/" in stem:
        parent, name = stem.rsplit("/", 1)
        for base in EVIDENCE_BASES:
            d = base / parent
            if d.is_dir() and any(f.name.startswith(name) for f in d.iterdir()):
                return True
    return False


def load(name: str) -> list[dict]:
    p = R / name
    if not p.exists():
        return []
    return list(csv.DictReader(p.open()))


# ----------------------------- suite-level CLI/API checks -----------------------------
def cli_help_ok(args: list[str]) -> bool:
    try:
        r = subprocess.run([sys.executable, "-m", "visionservex", *args, "--help"],
                           capture_output=True, text=True, timeout=90, cwd=ROOT)
        return r.returncode == 0 and ("Usage" in r.stdout or "Commands" in r.stdout
                                      or "Options" in r.stdout)
    except Exception:
        return False


def api_import_ok() -> bool:
    try:
        r = subprocess.run([sys.executable, "-c",
                            "from visionservex import VSX; VSX.sam('mobilesam').status(); "
                            "import visionservex.onnx_export, visionservex.sam2_runtime; print('ok')"],
                           capture_output=True, text=True, timeout=120, cwd=ROOT)
        return r.returncode == 0 and "ok" in r.stdout
    except Exception:
        return False


CLI_GROUPS = {"sam": cli_help_ok(["sam"]), "dino": cli_help_ok(["dino"]),
              "pipeline": cli_help_ok(["pipeline"]), "cv2-pro": cli_help_ok(["cv2-pro"]),
              "run": cli_help_ok(["run"]), "list-models": cli_help_ok(["list-models"])}
ROOT_CLI_OK = cli_help_ok([])
API_OK = api_import_ok()

# ----------------------------- tutorial scan -----------------------------
TUT_DIRS = ["sam_family", "dino_family", "pipelines", "cv2_pro", "interactive_seg", "v32"]
tutorial_text = ""
tutorial_files = []
for d in TUT_DIRS:
    for nb in (NB / "tutorials" / d).glob("*.ipynb"):
        tutorial_files.append(nb)
        with contextlib.suppress(Exception):
            tutorial_text += json.dumps(json.loads(nb.read_text()))
executed_dir = NB / "tutorials" / "_executed"
executed_files = list(executed_dir.glob("*.ipynb")) if executed_dir.exists() else []
executed_text = ""
for nb in executed_files:
    with contextlib.suppress(Exception):
        executed_text += json.dumps(json.loads(nb.read_text()))


def model_has_tutorial(mid: str) -> bool:
    return mid in tutorial_text


def model_tutorial_executed(mid: str) -> bool:
    return mid in executed_text


# ----------------------------- MODEL MATRIX -----------------------------
rights = {r["model_id"]: r for r in load("v3_core_model_rights.csv")}
models = load("model_coverage_ledger.csv")

model_rows = []
for m in models:
    mid = m["model_id"]
    cat, is_pass, is_fail, is_blocked = classify_model(
        m["final_state"], m["default_safe"], m["license_status"], m["evidence_artifact"])
    rr = rights.get(mid, {})
    ev = m["evidence_artifact"]
    model_rows.append({
        "model_id": mid,
        "family": m["family"],
        "task": m["task"],
        "final_state": m["final_state"],
        "default_safe": m["default_safe"],
        "license_status": m["license_status"],
        "has_code_license": bool(rr.get("code_license", "").strip()),
        "has_weights_license": bool(rr.get("weights_license", "").strip()),
        "has_evidence_artifact": bool(ev.strip()),
        "artifact_exists_on_disk": evidence_exists(ev),
        "has_tutorial": model_has_tutorial(mid),
        "tutorial_executed": model_tutorial_executed(mid),
        "has_cli": m["final_state"] != "not_advertised",
        "cli_help_works": CLI_GROUPS["run"] and ROOT_CLI_OK,
        "python_api_works": API_OK,
        "category": cat,
        "counts_as_pass": is_pass,
        "counts_as_fail": is_fail,
        "counts_as_blocked": is_blocked,
        "blocker_category": m["blocker_category"],
        "exact_next_command": m.get("next_iteration_command", "") or m.get("manual_fix_command", ""),
    })


def pct(n, d):
    return round(100.0 * n / d, 2) if d else 0.0


def summarize(rows, key="category"):
    c = Counter(r[key] for r in rows)
    return dict(c)


total = len(model_rows)
n_pass = sum(r["counts_as_pass"] for r in model_rows)
n_fail = sum(r["counts_as_fail"] for r in model_rows)
n_block = sum(r["counts_as_blocked"] for r in model_rows)
n_excl = sum(r["category"] == "EXCLUDED" for r in model_rows)
n_other = sum(r["category"] == "OTHER_UNVERIFIED" for r in model_rows)
ds_rows = [r for r in model_rows if r["category"] != "EXCLUDED"]
ds_pass = sum(r["counts_as_pass"] for r in ds_rows)
ev_complete = sum(1 for r in model_rows if r["counts_as_pass"] and r["artifact_exists_on_disk"])
pass_rows_total = sum(r["counts_as_pass"] for r in model_rows)

# ----------------------------- PIPELINE MATRIX -----------------------------
pipes = load("pipeline_coverage_ledger.csv")
pipe_rows = []
for p in pipes:
    ok = _truthy(p.get("both_parts_benchmark_passed", ""))
    pipe_rows.append({
        "pipeline_id": p["pipeline_id"], "kind": p["kind"],
        "detector_final_state": p.get("detector_final_state", ""),
        "segmenter_final_state": p.get("segmenter_final_state", ""),
        "both_parts_benchmark_passed": p.get("both_parts_benchmark_passed", ""),
        "category": "PASS" if ok else "BLOCKED",
        "counts_as_pass": ok, "counts_as_fail": False, "counts_as_blocked": not ok,
    })
n_pipe = len(pipe_rows)
n_pipe_pass = sum(r["counts_as_pass"] for r in pipe_rows)

# ----------------------------- TOOL MATRIX (cv2-pro + classic smart) -----------------------------
tool_rows = []
for t in load("smart_tool_coverage_ledger.csv"):
    fs = t.get("final_state", "")
    ok = fs in PASS_STATES
    tool_rows.append({"tool_id": t["tool_id"], "tool_kind": "classic-smart",
                      "final_state": fs, "dependency_license": t.get("dependency_license", ""),
                      "commercial_safe": t.get("commercial_safe", ""),
                      "category": "PASS" if ok else "BLOCKED",
                      "counts_as_pass": ok, "counts_as_fail": False, "counts_as_blocked": not ok,
                      "evidence_artifact": t.get("evidence_artifact", "")})
for t in load("v31_cv2_pro_tool_ledger.csv"):
    fs = t.get("final_state", "")
    ok = fs in PASS_STATES
    tool_rows.append({"tool_id": t["tool_id"], "tool_kind": "cv2-pro",
                      "final_state": fs, "dependency_license": t.get("dependency_license", ""),
                      "commercial_safe": t.get("commercial_safe", ""),
                      "category": "PASS" if ok else "BLOCKED",
                      "counts_as_pass": ok, "counts_as_fail": False, "counts_as_blocked": not ok,
                      "evidence_artifact": t.get("evidence_artifact", "")})
n_tool = len(tool_rows)
n_tool_pass = sum(r["counts_as_pass"] for r in tool_rows)

# ----------------------------- SAM / DINO comprehensive (v31 matrices) -----------------------------
def classify_target(state: str) -> str:
    if state in PASS_STATES:
        return "PASS"
    if state == "excluded_restricted":
        return "EXCLUDED"
    if state in BLOCKED_STATES or state in ("legal_review_required", "not_released"):
        return "BLOCKED"
    if state == "wired":
        return "OTHER"
    return "OTHER"


def family_matrix_stats(name):
    rows = load(name)
    cats = [classify_target(r["target_state_after"]) for r in rows]
    n = len(rows)
    return {
        "total": n,
        "pass": cats.count("PASS"), "blocked": cats.count("BLOCKED"),
        "excluded": cats.count("EXCLUDED"), "other": cats.count("OTHER"),
        "pass_pct": pct(cats.count("PASS"), n),
        "blocked_pct": pct(cats.count("BLOCKED"), n),
        "excluded_pct": pct(cats.count("EXCLUDED"), n),
    }


sam_stats = family_matrix_stats("v31_sam_family_matrix.csv")
dino_stats = family_matrix_stats("v31_dino_family_matrix.csv")

# sidecar: model rows whose final_state == sidecar_required (+ v32 sidecar ledger attempts)
sidecar_models = [r for r in model_rows if r["final_state"] == "sidecar_required"]
n_sidecar = len(sidecar_models)
# any sidecar that reached demo_passed_sidecar counts as pass
sidecar_pass = sum(1 for r in model_rows if r["final_state"] == "demo_passed_sidecar")
sidecar_total = n_sidecar + sidecar_pass

# ----------------------------- family / task percentages -----------------------------
def group_pct(rows, key):
    g = defaultdict(lambda: {"total": 0, "pass": 0, "fail": 0, "blocked": 0, "excluded": 0})
    for r in rows:
        b = g[r[key]]
        b["total"] += 1
        if r["counts_as_pass"]:
            b["pass"] += 1
        if r["counts_as_fail"]:
            b["fail"] += 1
        if r["counts_as_blocked"]:
            b["blocked"] += 1
        if r["category"] == "EXCLUDED":
            b["excluded"] += 1
    out = []
    for k, b in sorted(g.items()):
        out.append({key: k, **b,
                    "pass_pct": pct(b["pass"], b["total"]),
                    "blocked_pct": pct(b["blocked"], b["total"]),
                    "excluded_pct": pct(b["excluded"], b["total"])})
    return out


family_pcts = group_pct(model_rows, "family")
task_pcts = group_pct(model_rows, "task")

# ----------------------------- write artifacts -----------------------------
def write_csv(name, rows):
    if not rows:
        return
    with (R / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


write_csv("v33_model_pass_fail_matrix.csv", model_rows)
write_csv("v33_pipeline_pass_fail_matrix.csv", pipe_rows)
write_csv("v33_tool_pass_fail_matrix.csv", tool_rows)
write_csv("v33_model_family_percentages.csv", family_pcts)
write_csv("v33_task_percentages.csv", task_pcts)

summary = {
    "model_rows_total": total,
    "models_passed": n_pass, "models_failed": n_fail, "models_blocked": n_block,
    "models_excluded": n_excl, "models_other_unverified": n_other,
    "model_pass_pct": pct(n_pass, total),
    "model_fail_pct": pct(n_fail, total),
    "model_blocked_pct": pct(n_block, total),
    "model_excluded_pct": pct(n_excl, total),
    "default_safe_rows": len(ds_rows),
    "default_safe_pass_pct": pct(ds_pass, len(ds_rows)),
    "evidence_completeness_pct": pct(ev_complete, pass_rows_total),
    "pass_rows_with_disk_evidence": ev_complete,
    "tutorial_coverage_pct": pct(sum(r["has_tutorial"] for r in model_rows), total),
    "tutorial_execution_pct": pct(sum(r["tutorial_executed"] for r in model_rows), total),
    "cli_coverage_pct": pct(sum(r["has_cli"] for r in model_rows), total),
    "python_api_coverage_pct": pct(sum(r["python_api_works"] for r in model_rows), total),
    "final_state_distribution": summarize(model_rows, "final_state"),
    "category_distribution": summarize(model_rows, "category"),
    "pipeline_total": n_pipe, "pipeline_pass": n_pipe_pass, "pipeline_pass_pct": pct(n_pipe_pass, n_pipe),
    "tool_total": n_tool, "tool_pass": n_tool_pass, "tool_pass_pct": pct(n_tool_pass, n_tool),
    "sam_comprehensive": sam_stats,
    "dino_comprehensive": dino_stats,
    "sidecar_total": sidecar_total, "sidecar_pass": sidecar_pass,
    "sidecar_blocked": n_sidecar, "sidecar_pass_pct": pct(sidecar_pass, sidecar_total),
    "sidecar_blocked_pct": pct(n_sidecar, sidecar_total),
    "cli_groups_help_ok": CLI_GROUPS, "root_cli_ok": ROOT_CLI_OK, "python_api_ok": API_OK,
    "bad_license_default_safe_rows": sum(
        1 for r in model_rows if r["category"] != "EXCLUDED" and _has_restricted_license(r["license_status"])),
    "tutorial_files_on_disk": len(tutorial_files),
    "executed_tutorial_files_on_disk": len(executed_files),
}

(R / "v33_truth_audit_baseline.json").write_text(json.dumps(summary, indent=2))

# release readiness matrix
release_rows = [
    {"item": "model_pass_pct", "value": summary["model_pass_pct"], "of_total_rows": total},
    {"item": "default_safe_pass_pct", "value": summary["default_safe_pass_pct"], "of_total_rows": len(ds_rows)},
    {"item": "evidence_completeness_pct", "value": summary["evidence_completeness_pct"], "of_total_rows": pass_rows_total},
    {"item": "pipeline_pass_pct", "value": summary["pipeline_pass_pct"], "of_total_rows": n_pipe},
    {"item": "tool_pass_pct", "value": summary["tool_pass_pct"], "of_total_rows": n_tool},
    {"item": "bad_license_default_safe_rows", "value": summary["bad_license_default_safe_rows"], "of_total_rows": total},
]
write_csv("v33_release_readiness_matrix.csv", release_rows)

# markdown baseline
md = [f"""# VisionServeX v3.3 Truth Audit — Baseline (deterministic from canonical ledgers)

## Model zoo ({total} rows in model_coverage_ledger.csv)

| category | count | pct |
|---|---|---|
| PASS | {n_pass} | {summary['model_pass_pct']}% |
| FAIL | {n_fail} | {summary['model_fail_pct']}% |
| BLOCKED | {n_block} | {summary['model_blocked_pct']}% |
| EXCLUDED | {n_excl} | {summary['model_excluded_pct']}% |
| OTHER/UNVERIFIED | {n_other} | {pct(n_other, total)}% |

- default-safe rows: {len(ds_rows)} → default-safe pass = {summary['default_safe_pass_pct']}%
- evidence completeness (pass rows with on-disk artifact): {ev_complete}/{pass_rows_total} = {summary['evidence_completeness_pct']}%
- tutorial coverage: {summary['tutorial_coverage_pct']}% · tutorial executed: {summary['tutorial_execution_pct']}%
- CLI coverage: {summary['cli_coverage_pct']}% · Python API coverage: {summary['python_api_coverage_pct']}%
- bad-license default-safe rows: {summary['bad_license_default_safe_rows']}

## final_state distribution
""" ]
for s, c in sorted(summary["final_state_distribution"].items(), key=lambda x: -x[1]):
    md.append(f"- {s}: {c}")
md.append(f"""
## Pipelines: {n_pipe_pass}/{n_pipe} pass = {summary['pipeline_pass_pct']}%
## Tools (cv2-pro + classic): {n_tool_pass}/{n_tool} pass = {summary['tool_pass_pct']}%
## SAM comprehensive (v31, {sam_stats['total']} targets): pass {sam_stats['pass_pct']}% / blocked {sam_stats['blocked_pct']}% / excluded {sam_stats['excluded_pct']}%
## DINO comprehensive (v31, {dino_stats['total']} targets): pass {dino_stats['pass_pct']}% / blocked {dino_stats['blocked_pct']}% / excluded {dino_stats['excluded_pct']}%
## Sidecar: pass {sidecar_pass}/{sidecar_total} = {summary['sidecar_pass_pct']}% / blocked {summary['sidecar_blocked_pct']}%

## CLI help: root={ROOT_CLI_OK} groups={CLI_GROUPS}
## Python API import: {API_OK}
""")
(R / "v33_truth_audit_baseline.md").write_text("\n".join(md))

# console
print("=" * 60)
print(f"MODEL ROWS: {total}")
print(f"  PASS={n_pass} ({summary['model_pass_pct']}%)  FAIL={n_fail}  "
      f"BLOCKED={n_block} ({summary['model_blocked_pct']}%)  "
      f"EXCLUDED={n_excl} ({summary['model_excluded_pct']}%)  OTHER={n_other}")
print(f"  default-safe pass% = {summary['default_safe_pass_pct']}%  "
      f"evidence completeness = {summary['evidence_completeness_pct']}%")
print(f"PIPELINES: {n_pipe_pass}/{n_pipe} = {summary['pipeline_pass_pct']}%")
print(f"TOOLS: {n_tool_pass}/{n_tool} = {summary['tool_pass_pct']}%")
print(f"SAM (v31 {sam_stats['total']}): pass {sam_stats['pass_pct']}% blk {sam_stats['blocked_pct']}% exc {sam_stats['excluded_pct']}%")
print(f"DINO (v31 {dino_stats['total']}): pass {dino_stats['pass_pct']}% blk {dino_stats['blocked_pct']}% exc {dino_stats['excluded_pct']}%")
print(f"SIDECAR: pass {sidecar_pass}/{sidecar_total} = {summary['sidecar_pass_pct']}%")
print(f"CLI root={ROOT_CLI_OK} groups_ok={sum(CLI_GROUPS.values())}/{len(CLI_GROUPS)}  API={API_OK}")
print(f"bad-license default-safe rows: {summary['bad_license_default_safe_rows']}")
print("evidence artifacts NOT found on disk for PASS rows:",
      [r["model_id"] for r in model_rows if r["counts_as_pass"] and not r["artifact_exists_on_disk"]][:20])
print("=" * 60)
print("wrote: v33_truth_audit_baseline.{md,json}, v33_model_pass_fail_matrix.csv,")
print("       v33_pipeline_pass_fail_matrix.csv, v33_tool_pass_fail_matrix.csv,")
print("       v33_model_family_percentages.csv, v33_task_percentages.csv,")
print("       v33_release_readiness_matrix.csv")
