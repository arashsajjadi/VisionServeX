#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Parse a pytest junit XML into v3.8 summary artifacts.

    python scripts/v38_parse_pytest.py <junit.xml> [<prefix>]

Writes (under notebook/99_final_report/reports/):
    <prefix>_summary.json, <prefix>_failed_tests.csv, <prefix>_test_execution_matrix.csv
"""

from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REPORTS = Path("notebook/99_final_report/reports")


def main() -> int:
    xml_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORTS / "v38_pytest.xml"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "v38_pytest"
    if not xml_path.exists():
        print(f"missing junit xml: {xml_path}")
        return 1
    root = ET.parse(xml_path).getroot()
    suites = root.findall(".//testsuite") or [root]
    total = passed = failed = errors = skipped = 0
    failed_rows = []
    matrix_rows = []
    per_file = Counter()
    per_file_fail = Counter()
    for suite in suites:
        for case in suite.findall("testcase"):
            total += 1
            fname = case.get("classname", "").replace(".", "/")
            name = case.get("name", "")
            status = "passed"
            if case.find("failure") is not None:
                status = "failed"
                failed += 1
                failed_rows.append({"file": fname, "test": name,
                                    "message": (case.find("failure").get("message", "") or "")[:300]})
            elif case.find("error") is not None:
                status = "error"
                errors += 1
                failed_rows.append({"file": fname, "test": name,
                                    "message": (case.find("error").get("message", "") or "")[:300]})
            elif case.find("skipped") is not None:
                status = "skipped"
                skipped += 1
            else:
                passed += 1
            key = fname.split("/")[-1] if fname else name
            per_file[key] += 1
            if status in ("failed", "error"):
                per_file_fail[key] += 1
            matrix_rows.append({"file": key, "test": name, "status": status,
                                "time_s": case.get("time", "")})

    summary = {"total": total, "passed": passed, "failed": failed,
               "errors": errors, "skipped": skipped,
               "pass_rate": round(passed / max(total - skipped, 1), 4)}
    (REPORTS / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    with (REPORTS / f"{prefix}_failed_tests.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "test", "message"])
        w.writeheader()
        w.writerows(failed_rows)
    with (REPORTS / f"{prefix}_test_execution_matrix.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "test", "status", "time_s"])
        w.writeheader()
        w.writerows(matrix_rows)
    print(json.dumps(summary, indent=2))
    print(f"wrote {prefix}_summary.json / _failed_tests.csv / _test_execution_matrix.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
