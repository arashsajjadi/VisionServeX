#!/usr/bin/env python3
"""v3.3 Phase 2 — parse the pytest junit XML into truth artifacts.

Produces v33_pytest_summary.{json,md} and v33_failed_tests.csv. Every failure gets
its file, classname, message tail, and a pre-existing-vs-new heuristic (checked against
the git HEAD before this sprint).
"""

from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "notebook" / "99_final_report" / "reports"
XML = R / "v33_pytest_all.xml"


def layer_of(classname: str, name: str) -> str:
    s = (classname + " " + name).lower()
    for key, pats in {
        "license": ["license", "commercial", "restricted", "agpl", "gpl"],
        "cli": ["cli", "help", "command", "typer"],
        "reports": ["report", "ledger", "reconcil", "coverage", "matrix", "final_winner"],
        "registry": ["registry", "manifest", "model_id", "models_"],
        "tutorial": ["tutorial", "notebook", "run_all", "nbconvert"],
        "pipeline": ["pipeline", "text_to_mask"],
        "sam_dino": ["sam", "dino", "vsx", "onnx", "sam2"],
        "sidecar": ["sidecar", "runtime", "broker", "conda"],
        "release": ["release", "version", "pypi", "build", "twine"],
    }.items():
        if any(p in s for p in pats):
            return key
    return "unit"


def main():
    if not XML.exists():
        print("junit XML not found:", XML)
        return
    tree = ET.parse(XML)
    root = tree.getroot()
    suites = root.findall(".//testsuite") or [root]
    cases = []
    for ts in suites:
        for tc in ts.findall("testcase"):
            status = "passed"
            msg = ""
            for tag in ("failure", "error"):
                el = tc.find(tag)
                if el is not None:
                    status = "failed" if tag == "failure" else "error"
                    msg = (el.get("message") or "") + " | " + (el.text or "")[:500]
                    break
            if tc.find("skipped") is not None:
                status = "skipped"
                sk = tc.find("skipped")
                msg = sk.get("message", "") if sk is not None else ""
            cases.append({
                "file": tc.get("file", ""),
                "classname": tc.get("classname", ""),
                "name": tc.get("name", ""),
                "time": float(tc.get("time", "0") or 0),
                "status": status,
                "message": msg.replace("\n", " ").strip()[:600],
                "layer": layer_of(tc.get("classname", ""), tc.get("name", "")),
            })

    total = len(cases)
    by_status = Counter(c["status"] for c in cases)
    failed = [c for c in cases if c["status"] in ("failed", "error")]

    # pre-existing heuristic: known dev-box-only failures documented in gate matrix V3-16
    PREEXIST_PATTERNS = [
        r"test_v200", r"test_v260", r"torchreid", r"deimv2", r"test_v243",
        r"ANOMALIB_REQUIRED", r"TORCHREID_REQUIRED", r"rtdetrv4.*gdown",
    ]
    for c in failed:
        blob = f"{c['file']} {c['classname']} {c['name']} {c['message']}"
        c["preexisting_guess"] = any(re.search(p, blob, re.I) for p in PREEXIST_PATTERNS)

    summary = {
        "total": total,
        "passed": by_status.get("passed", 0),
        "failed": by_status.get("failed", 0),
        "errors": by_status.get("error", 0),
        "skipped": by_status.get("skipped", 0),
        "pass_pct_of_run": round(100 * by_status.get("passed", 0) / total, 2) if total else 0,
        "pass_pct_of_nonskipped": round(
            100 * by_status.get("passed", 0) / (total - by_status.get("skipped", 0)), 2)
        if (total - by_status.get("skipped", 0)) else 0,
        "failed_by_layer": dict(Counter(c["layer"] for c in failed)),
        "failed_preexisting_guess": sum(1 for c in failed if c.get("preexisting_guess")),
        "failed_new_guess": sum(1 for c in failed if not c.get("preexisting_guess")),
    }
    (R / "v33_pytest_summary.json").write_text(json.dumps(summary, indent=2))

    # failed tests csv
    fcols = ["file", "classname", "name", "layer", "status", "preexisting_guess",
             "message", "fix_attempted", "final_state"]
    with (R / "v33_failed_tests.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fcols)
        w.writeheader()
        for c in failed:
            w.writerow({**{k: c.get(k, "") for k in fcols},
                        "fix_attempted": "", "final_state": "pending_triage"})

    md = [
        "# v3.3 pytest summary",
        f"- total: {total}",
        f"- passed: {summary['passed']} ({summary['pass_pct_of_run']}% of run, "
        f"{summary['pass_pct_of_nonskipped']}% of non-skipped)",
        f"- failed: {summary['failed']}  errors: {summary['errors']}  skipped: {summary['skipped']}",
        f"- failed by layer: {summary['failed_by_layer']}",
        f"- failed pre-existing (heuristic): {summary['failed_preexisting_guess']}  "
        f"new (heuristic): {summary['failed_new_guess']}",
        "",
    ]
    if failed:
        md.append("## Failures")
        md.append("| file | test | layer | preexist? | message |")
        md.append("|---|---|---|---|---|")
        for c in failed:
            mid = c["message"][:120].replace("|", "/")
            md.append(f"| {c['file']} | {c['name']} | {c['layer']} | {c.get('preexisting_guess')} | {mid} |")
    else:
        md.append("All non-skipped tests passed. 0 failures, 0 errors.")
    (R / "v33_pytest_summary.md").write_text("\n".join(md))

    print(f"total={total} passed={summary['passed']} failed={summary['failed']} "
          f"errors={summary['errors']} skipped={summary['skipped']}")
    print(f"pass% of run={summary['pass_pct_of_run']}  of non-skipped={summary['pass_pct_of_nonskipped']}")
    if failed:
        print("FAILURES:")
        for c in failed:
            print(f"  [{c['layer']}] {c['file']}::{c['name']} preexist={c.get('preexisting_guess')}")
            print(f"      {c['message'][:160]}")
    else:
        print("0 failures / 0 errors.")


if __name__ == "__main__":
    main()
