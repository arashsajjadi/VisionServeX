#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Generate the v3.8 license policy matrix CSV + report from the canonical policy.

Single source of truth: ``visionservex.licensing.policy``. Run from the repo
root::

    python scripts/v38_generate_license_matrix.py

Writes (under notebook/99_final_report/reports/):
    v38_license_policy_matrix.csv
    v38_license_policy_report.md
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from visionservex.licensing import policy as P
from visionservex.licensing.policy import MATRIX_COLUMNS

REPORTS = Path("notebook/99_final_report/reports")
CSV_PATH = REPORTS / "v38_license_policy_matrix.csv"
MD_PATH = REPORTS / "v38_license_policy_report.md"
JSON_PATH = REPORTS / "v38_license_policy_matrix.json"

_HARD_RULES = [
    "A Hugging Face token does NOT grant redistribution rights.",
    "Gated models are never packaged into PyPI / GitHub / Docker (can_ship_weights=False for every row).",
    "Non-commercial models never run in production mode and are never default_safe.",
    "AGPL / enterprise models never enter the default_safe core.",
    "API-only models are never counted as local models (is_local=False).",
    "legal_review models are never commercial_safe until the review is resolved.",
    "Code license, weights license, and dataset/pretraining risk are tracked separately.",
]


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = P.matrix_rows()

    # CSV
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(MATRIX_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # JSON (machine-readable)
    JSON_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # Report
    counts = Counter(r["final_policy"] for r in rows)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[r["final_policy"]].append(r)

    lines: list[str] = []
    lines.append("# VisionServeX v3.8 — License Policy Matrix Report")
    lines.append("")
    lines.append("Generated from `visionservex.licensing.policy` (single source of truth). "
                 "The CLI, Python API, tests, notebooks, and docs all read this same table.")
    lines.append("")
    lines.append(f"**Total models classified: {len(rows)}**")
    lines.append("")
    lines.append("## Summary by final policy")
    lines.append("")
    lines.append("| final_policy | count |")
    lines.append("|---|---|")
    for fp in P.FINAL_POLICIES:
        lines.append(f"| `{fp}` | {counts.get(fp, 0)} |")
    lines.append(f"| **total** | **{len(rows)}** |")
    lines.append("")

    lines.append("## Hard rules (enforced in code + tests)")
    lines.append("")
    for rule in _HARD_RULES:
        lines.append(f"- {rule}")
    lines.append("")

    lines.append("## Warning texts")
    lines.append("")
    for key, txt in P.WARNING_TEXTS.items():
        lines.append(f"- **{key}**: {txt}")
    lines.append("")

    lines.append("## Models by policy")
    lines.append("")
    for fp in P.FINAL_POLICIES:
        bucket = by_bucket.get(fp, [])
        if not bucket:
            continue
        lines.append(f"### `{fp}` ({len(bucket)})")
        lines.append("")
        lines.append("| model_id | family | code license | weights license | gated | commercial_safe | production | next command |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in bucket:
            lines.append(
                f"| `{r['model_id']}` | {r['family']} | {r['code_license']} | "
                f"{r['weights_license']} | {r['gated']} | {r['commercial_safe']} | "
                f"{r['production_allowed']} | `{r['exact_next_command']}` |"
            )
        lines.append("")

    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {CSV_PATH} ({len(rows)} rows)")
    print(f"wrote {JSON_PATH}")
    print(f"wrote {MD_PATH}")
    print("counts:", dict(counts))


if __name__ == "__main__":
    main()
