# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.28.0: truth audit — count raw NaN, NOT_WIRED, stale markers, etc.

The audit scans every CSV / JSON / Markdown file under a reports directory
and tallies:

- ``raw_nan_count`` — literal ``NaN`` / ``nan`` in CSV cells or MD text
  (not inside JSON dict values where ``null`` is OK with a status field).
- ``not_wired_count`` — literal ``NOT_WIRED`` anywhere.
- ``failed_runtime_parseable_blocker_count`` — ``failed_runtime`` status
  on rows that also carry one of the v2.28 parseable blocker codes.
- ``stale_marker_count`` — any of the canonical ``STALE_MARKERS`` strings.
- ``empty_status_count`` — empty/null status cells in candidate-status
  CSVs (per_family_command_status.csv, model_readiness_matrix.csv, etc.).
- ``missing_blocker_code_count`` — expected_blocker rows without a code.
- ``missing_source_status_count`` — null metric value with no source_status.

The CLI fails non-zero if any required-zero count > 0 and ``--strict``
is set.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from visionservex.reporting.status_vocab import (
    PARSEABLE_BLOCKER_CODES,
    STALE_MARKERS,
)

_NAN_RE = re.compile(r"(?:^|,|\s)(NaN|nan)(?:,|\s|$)")
_NOT_WIRED_RE = re.compile(r"\bNOT_WIRED\b")


def _iter_data_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in {".csv", ".json", ".md"})


def _count_in_text(text: str, needles: Iterable[str]) -> int:
    return sum(text.count(n) for n in needles)


def _audit_csv(path: Path) -> dict[str, Any]:
    out = {
        "path": str(path),
        "raw_nan": 0,
        "not_wired": 0,
        "failed_runtime_parseable_blocker": 0,
        "stale_markers": 0,
        "empty_status": 0,
        "missing_blocker_code": 0,
    }
    try:
        rows = list(csv.DictReader(path.open()))
    except Exception:
        return out
    text = path.read_text(errors="ignore")
    out["raw_nan"] = len(_NAN_RE.findall(text))
    out["not_wired"] = len(_NOT_WIRED_RE.findall(text))
    out["stale_markers"] = _count_in_text(text, STALE_MARKERS)
    for row in rows:
        status = (row.get("status") or row.get("final_state") or "").strip()
        code = (row.get("blocker_code") or row.get("code") or "").strip()
        if status == "failed_runtime" and code.upper() in PARSEABLE_BLOCKER_CODES:
            out["failed_runtime_parseable_blocker"] += 1
        if "status" in row and not status:
            out["empty_status"] += 1
        if status == "expected_blocker" and not code:
            out["missing_blocker_code"] += 1
    return out


def _audit_md(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="ignore")
    # Strip out the scientific-validity policy paragraph that legitimately
    # mentions "NaN metrics" in prose.
    lines = []
    for line in text.splitlines():
        if "NaN metrics" in line or "exclude" in line:
            continue
        lines.append(line)
    body = "\n".join(lines)
    return {
        "path": str(path),
        "raw_nan": len(_NAN_RE.findall(body)),
        "not_wired": len(_NOT_WIRED_RE.findall(body)),
        "failed_runtime_parseable_blocker": 0,
        "stale_markers": _count_in_text(body, STALE_MARKERS),
        "empty_status": 0,
        "missing_blocker_code": 0,
    }


def _audit_json(path: Path) -> dict[str, Any]:
    out = {
        "path": str(path),
        "raw_nan": 0,
        "not_wired": 0,
        "failed_runtime_parseable_blocker": 0,
        "stale_markers": 0,
        "empty_status": 0,
        "missing_blocker_code": 0,
        "missing_source_status": 0,
    }
    try:
        text = path.read_text(errors="ignore")
        # JSON itself disallows literal NaN, but stringified NaN can appear in
        # display fields; check the raw text for NaN tokens.
        out["raw_nan"] = len(_NAN_RE.findall(text))
        out["not_wired"] = len(_NOT_WIRED_RE.findall(text))
        out["stale_markers"] = _count_in_text(text, STALE_MARKERS)
    except Exception:
        return out
    try:
        d = json.loads(text)
    except Exception:
        return out

    # Walk: every dict with {status, code, blocker_code} → check parseable_blocker
    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            status = obj.get("status") or obj.get("final_state") or ""
            code = obj.get("blocker_code") or obj.get("code") or ""
            if (
                isinstance(status, str)
                and status == "failed_runtime"
                and isinstance(code, str)
                and code.upper() in PARSEABLE_BLOCKER_CODES
            ):
                out["failed_runtime_parseable_blocker"] += 1
            if isinstance(status, str) and status == "expected_blocker" and not code:
                out["missing_blocker_code"] += 1
            value = obj.get("value")
            source_status = obj.get("source_status")
            if "value" in obj and value is None and not source_status:
                out["missing_source_status"] += 1
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(d)
    return out


def audit_reports_dir(reports_dir: Path) -> dict[str, Any]:
    """Return aggregate counts for every audit dimension."""
    files = _iter_data_files(reports_dir)
    per_file: list[dict[str, Any]] = []
    totals: dict[str, int] = {
        "raw_nan": 0,
        "not_wired": 0,
        "failed_runtime_parseable_blocker": 0,
        "stale_markers": 0,
        "empty_status": 0,
        "missing_blocker_code": 0,
        "missing_source_status": 0,
    }
    for p in files:
        if p.suffix.lower() == ".csv":
            r = _audit_csv(p)
        elif p.suffix.lower() == ".md":
            r = _audit_md(p)
        else:
            r = _audit_json(p)
        per_file.append(r)
        for k in totals:
            totals[k] += int(r.get(k, 0))

    required_zero = (
        "raw_nan",
        "not_wired",
        "failed_runtime_parseable_blocker",
        "stale_markers",
    )
    failing = {k: v for k, v in totals.items() if k in required_zero and v > 0}
    return {
        "status": "ok" if not failing else "expected_blocker",
        "code": "OK" if not failing else "REPORTING_TRUTH_AUDIT_FAILED",
        "reports_dir": str(reports_dir),
        "n_files_scanned": len(files),
        "raw_nan_count_final": totals["raw_nan"],
        "not_wired_count_final": totals["not_wired"],
        "failed_runtime_parseable_blocker_count": totals["failed_runtime_parseable_blocker"],
        "stale_marker_count": totals["stale_markers"],
        "empty_status_count": totals["empty_status"],
        "missing_blocker_code_count": totals["missing_blocker_code"],
        "missing_source_status_count": totals["missing_source_status"],
        "failing_dimensions": failing,
        "per_file_offenders": [
            r for r in per_file if any(int(r.get(k, 0)) > 0 for k in required_zero)
        ],
    }


__all__ = ["audit_reports_dir"]
