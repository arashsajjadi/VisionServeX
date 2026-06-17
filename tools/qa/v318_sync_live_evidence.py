# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Bake the committed v3.18 live matrices into ``readiness/live_evidence.py``.

The matrices under ``docs/qa/v318_full_model_truth/`` are the *evidence*; the
frozensets/dict in ``live_evidence.py`` are the *conclusions* the package ships.
This tool regenerates the three GENERATED blocks from the matrices so the two can
never drift (``tests/test_v318_capability_truth_contract.py`` enforces it).

Run after every live matrix run::

    python tools/qa/v318_sync_live_evidence.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QA = REPO / "docs" / "qa" / "v318_full_model_truth"
TARGET = REPO / "src" / "visionservex" / "readiness" / "live_evidence.py"


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data.get("results", data) if isinstance(data, dict) else data


def classify_inference_blocker(row: dict) -> str:
    """Map a non-PASS inference row to its precise readiness blocker state."""
    blk = (row.get("blocker") or "").upper()
    et = row.get("error_type") or ""
    msg = (row.get("error_message") or "").lower()
    if blk == "OOM_BLOCKED" or "out of memory" in msg or "cannot allocate" in msg:
        return "OOM_BLOCKED"
    if blk == "WEIGHTS_DOWNLOAD_UNAVAILABLE" or et == "DownloadError" or "404" in msg:
        return "WEIGHTS_MISSING"
    if (
        blk == "DEPENDENCY_MISSING"
        or et in ("MissingDependencyError", "ImportError", "ModuleNotFoundError")
        or "incompatible with transformers" in msg
        or "no module named" in msg
    ):
        return "DEPENDENCY_MISSING"
    if blk == "TIMEOUT" or et == "Timeout":
        return "UPSTREAM_CRASH"
    if et == "TaskNotSupportedError":
        return "TASK_NOT_SUPPORTED"
    if et == "SchemaInvalid":
        return "UNKNOWN_REVIEW_REQUIRED"
    return "UPSTREAM_CRASH"


def _fmt_frozenset(ids: list[str]) -> str:
    if not ids:
        return "frozenset(set())"
    body = "\n".join(f'        "{i}",' for i in sorted(ids))
    return "frozenset(\n    {\n" + body + "\n    }\n)"


def _fmt_dict(d: dict[str, str]) -> str:
    if not d:
        return "{}"
    body = "\n".join(f'    "{k}": "{d[k]}",' for k in sorted(d))
    return "{\n" + body + "\n}"


def _replace_block(text: str, name: str, replacement: str) -> str:
    pat = re.compile(
        rf"(# >>> BEGIN GENERATED: {name}\n).*?(\n# <<< END GENERATED: {name})",
        re.DOTALL,
    )
    if not pat.search(text):
        raise SystemExit(f"marker block for {name} not found in {TARGET}")
    # Re-emit the assignment line + value inside the markers.
    if name == "LIVE_INFERENCE_VERIFIED":
        assign = f"LIVE_INFERENCE_VERIFIED: frozenset[str] = {replacement}"
    elif name == "LIVE_TRAIN_VERIFIED":
        assign = f"LIVE_TRAIN_VERIFIED: frozenset[str] = {replacement}"
    else:
        assign = f"LIVE_INFERENCE_FAILED: dict[str, str] = {replacement}"
    return pat.sub(lambda m: m.group(1) + assign + m.group(2), text)


def main() -> int:
    qa319 = REPO / "docs" / "qa" / "v319_operationalize_all_models"
    inf = _rows(QA / "live_inference_matrix.json")
    trn = _rows(QA / "live_train_lifecycle_matrix.json")
    # v3.19 operationalization matrices (additive — union with the v3.18 evidence).
    rfdetr_trn = _rows(qa319 / "rfdetr_live_train_matrix.json")
    inf319 = _rows(qa319 / "v319_inference_matrix.json")

    inf_pass = sorted(
        {
            r["model_id"]
            for r in (inf + inf319)
            if r.get("status") == "PASS" and r.get("live_verified")
        }
    )
    # A v3.19 PASS clears any earlier blocker for that model.
    promoted = set(inf_pass)
    inf_failed = {
        r["model_id"]: classify_inference_blocker(r)
        for r in (inf + inf319)
        if r.get("status") in ("FAIL", "SKIP_BLOCKED") and r["model_id"] not in promoted
    }
    trn_pass = sorted(
        {
            r["model_id"]
            for r in (trn + rfdetr_trn)
            if r.get("status") == "PASS" and r.get("live_verified")
        }
    )

    text = TARGET.read_text()
    text = _replace_block(text, "LIVE_INFERENCE_VERIFIED", _fmt_frozenset(inf_pass))
    text = _replace_block(text, "LIVE_TRAIN_VERIFIED", _fmt_frozenset(trn_pass))
    text = _replace_block(text, "LIVE_INFERENCE_FAILED", _fmt_dict(inf_failed))
    TARGET.write_text(text)

    print(
        f"[sync] live_inference PASS={len(inf_pass)} FAILED/BLOCKED={len(inf_failed)} "
        f"| live_train PASS={len(trn_pass)}"
    )
    print("[sync] inference blockers:", json.dumps(inf_failed, indent=0))
    print(f"[sync] wrote {TARGET.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
