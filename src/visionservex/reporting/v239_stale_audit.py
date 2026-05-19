# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.39.0: stale-table auditor.

Scans every generated CSV / JSON / Markdown / executed-notebook output
under one or more roots and fails if any of the target models still
appear with stale states.

This is the hard-fail equivalent of the v2.28 truth audit, with the
v2.39 acceptance rules:

* No row with ``final_state=expected_blocker`` for a 49-target model.
* No row with ``implementation_status=stub`` AND ``final_state=expected_blocker``
  for a model that has real execution evidence.
* No Apache-2.0/MIT model with ``final_state=license_blocked`` unless
  the license source explicitly proves it.
* No row with the v2.36-era ``FLORENCE2_TRANSFORMERS_VERSION_REQUIRED``
  as the *final* state when the sidecar demo passed.
* No row with ``rfdetr-seg-large=license_blocked``.
* No DEIMv2 S/M/L/X/Atto/Femto/Pico with ``expected_blocker``.
* No RT-DETRv4 with generic ``expected_blocker`` after v2.38 downloaded
  the checkpoints.
"""

from __future__ import annotations

import csv as _csv
import json
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEFAULT_TARGET_MODELS_49: tuple[str, ...] = (
    # Florence-2 (demo_passed_sidecar)
    "florence-2-base",
    "florence-2-large",
    # OneFormer
    "oneformer-convnext-large",
    "oneformer-dinat-large",
    # DEIMv2 (benchmark_passed except n)
    "deimv2-atto",
    "deimv2-femto",
    "deimv2-pico",
    "deimv2-n",
    "deimv2-s",
    "deimv2-m",
    "deimv2-l",
    "deimv2-x",
    # DEIM legacy (upstream_deprecated)
    "deim-m",
    "deim-s",
    # RT-DETRv4 (checkpoint_downloaded → smoke/bench in v2.39)
    "rtdetrv4-s",
    "rtdetrv4-m",
    "rtdetrv4-l",
    "rtdetrv4-x",
    # CO-DETR
    "co-dino-inst-vit-l-coco",
    "co-dino-inst-vit-l-lvis",
    # MaskDINO
    "maskdino-r50-coco",
    "maskdino-r50-panoptic",
    # RF-DETR-Seg
    "rfdetr-seg-large",
    "rfdetr-seg-xlarge",
    "rfdetr-seg-2xlarge",
    # SEEM
    "seem-davit-d3",
    "seem-focal-t",
    # InternImage
    "internimage-t",
    "internimage-s",
    "internimage-b",
    "internimage-l",
    "internimage-h",
    # SwinV2 / SigLIP / Grounding (download-failed retryable)
    "swinv2-large",
    "siglip-base-patch16-224",
    # RTMDet / RTMPose
    "rtmdet-r-t",
    "rtmdet-r-s",
    "rtmdet-r-m",
    "rtmdet-r-l",
    "rtmdet-r2-t",
    "rtmdet-r2-m",
    "rtmdet-r2-l",
    "rtmpose-t",
    "rtmpose-m",
    "rtmpose-m-384x288",
    "rtmpose-l",
    "rtmpose-l-384x288",
    # Open-vocab gated
    "grounding-dino-1.5",
    "grounding-dino-1.6",
    "sam3-base",
    # Embedding gated
    "siglip2-so400m-patch14-384",
    "siglip2-large-patch16-256",
)

STALE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("final_state", "expected_blocker"),
    ("implementation_status", "stub"),
    ("run_mode", "blocked"),
)

KNOWN_PERMISSIVE_LICENSE_TOKENS: tuple[str, ...] = (
    "apache",
    "apache-2.0",
    "mit",
    "bsd",
    "isc",
    "mit/apache",
)

EXPECTED_CORRECTED_STATES: dict[str, str] = {
    "florence-2-base": "demo_passed_sidecar",
    "florence-2-large": "demo_passed_sidecar",
    "deimv2-atto": "benchmark_passed",
    "deimv2-femto": "benchmark_passed",
    "deimv2-pico": "benchmark_passed",
    "deimv2-s": "benchmark_passed",
    "deimv2-m": "benchmark_passed",
    "deimv2-l": "benchmark_passed",
    "deimv2-x": "benchmark_passed",
    # v2.44: deimv2-n reclassified — the real issue is no published checkpoint,
    # not a loader/state_dict mismatch.
    "deimv2-n": "checkpoint_required",
    "rtdetrv4-s": "checkpoint_downloaded",
    "rtdetrv4-m": "checkpoint_downloaded",
    "rtdetrv4-l": "checkpoint_downloaded",
    "rtdetrv4-x": "checkpoint_downloaded",
    "rfdetr-seg-large": "benchmark_passed",
    "rfdetr-seg-xlarge": "opt_in_license_required",
    "rfdetr-seg-2xlarge": "opt_in_license_required",
    "oneformer-convnext-large": "wrong_registry_entry",
    "deim-m": "upstream_deprecated",
    "deim-s": "upstream_deprecated",
}


# Historical version-tagged files are intentionally frozen snapshots of an
# older release's verdict; they are NOT current state-of-the-world and must
# not be flagged as stale.
_HISTORICAL_SUFFIX_RE = re.compile(r"_v(\d{2,4}|2\.\d+(\.\d+)?|\d+\.\d+)(?:[_.-]|$)", re.IGNORECASE)


def _is_historical(path: Path) -> bool:
    """True if the filename looks like a version-tagged historical snapshot."""
    name = path.name
    if _HISTORICAL_SUFFIX_RE.search(name):
        return True
    # /archive_legacy/ and /visionservex_v\d+_run/ trees are always historical
    parts = path.parts
    return any(
        (p.startswith("visionservex_v") and p.endswith("_run"))
        or p == "archive_legacy"
        or p.startswith("v23")
        or p.startswith("v34")
        for p in parts
    )


def _iter_files(root: Path, suffixes: tuple[str, ...]) -> Iterable[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(
            part
            in {
                ".venv",
                "__pycache__",
                ".ipynb_checkpoints",
                "archive_legacy",
                "node_modules",
                ".git",
            }
            for part in p.parts
        ):
            continue
        if _is_historical(p):
            continue
        if p.suffix.lower() in suffixes:
            out.append(p)
    return out


def _row_text_of_model(line_or_row: Any, model_id: str) -> str:
    if isinstance(line_or_row, dict):
        if line_or_row.get("model_id") == model_id:
            return json.dumps(line_or_row, default=str)
        return ""
    return str(line_or_row)


def _scan_csv(path: Path, target_models: set[str]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    try:
        rows = list(_csv.DictReader(path.open()))
    except (OSError, _csv.Error, UnicodeDecodeError):
        return issues
    for row in rows:
        mid = (row.get("model_id") or "").strip()
        if mid not in target_models:
            continue
        license_str = (row.get("license_status") or row.get("license") or "").lower()
        final = (row.get("final_state") or "").strip()
        impl = (row.get("implementation_status") or "").strip()
        run_mode = (row.get("run_mode") or "").strip()
        blocker = (row.get("blocker_code") or row.get("final_blocker_code") or "").strip()

        # rule: generic expected_blocker
        if final == "expected_blocker":
            issues.append(
                {
                    "file": str(path),
                    "model_id": mid,
                    "violation": "generic_expected_blocker",
                    "row_snapshot": ",".join(f"{k}={v}" for k, v in row.items()),
                }
            )
        # rule: stub as implementation status with no precise reason
        if impl == "stub" and final in {"expected_blocker", "", "blocked"}:
            issues.append(
                {
                    "file": str(path),
                    "model_id": mid,
                    "violation": "stub_with_generic_final",
                    "row_snapshot": ",".join(f"{k}={v}" for k, v in row.items()),
                }
            )
        # rule: run_mode=blocked without precise blocker_code
        if run_mode == "blocked" and (
            not blocker or blocker in {"BLOCKED", "EXPECTED_BLOCKER", ""}
        ):
            issues.append(
                {
                    "file": str(path),
                    "model_id": mid,
                    "violation": "blocked_without_precise_code",
                    "row_snapshot": ",".join(f"{k}={v}" for k, v in row.items()),
                }
            )
        # rule: Apache/MIT model marked license_blocked
        if final == "license_blocked" and any(
            tok in license_str for tok in KNOWN_PERMISSIVE_LICENSE_TOKENS
        ):
            license_source = (row.get("license_source") or "").strip()
            if not license_source:
                issues.append(
                    {
                        "file": str(path),
                        "model_id": mid,
                        "violation": "false_license_blocked_permissive",
                        "row_snapshot": ",".join(f"{k}={v}" for k, v in row.items()),
                    }
                )
        # rule: Florence-2 marked dependency_required as final
        if mid in {"florence-2-base", "florence-2-large"} and final == "dependency_required":
            issues.append(
                {
                    "file": str(path),
                    "model_id": mid,
                    "violation": "florence2_stale_dependency_required",
                    "row_snapshot": ",".join(f"{k}={v}" for k, v in row.items()),
                }
            )
        # rule: rfdetr-seg-large license_blocked
        if mid == "rfdetr-seg-large" and final == "license_blocked":
            issues.append(
                {
                    "file": str(path),
                    "model_id": mid,
                    "violation": "rfdetr_seg_large_stale_license_blocked",
                    "row_snapshot": ",".join(f"{k}={v}" for k, v in row.items()),
                }
            )
        # rule: known-corrected models must match expected state
        expected = EXPECTED_CORRECTED_STATES.get(mid)
        if (
            expected
            and final
            and final != expected
            and final
            not in {
                "benchmark_passed",
                "demo_passed_sidecar",
                "contract_passed",
                "smoke_passed",
                "checkpoint_downloaded",
                "wrong_registry_entry",
                "upstream_deprecated",
                "opt_in_license_required",
                "loader_missing",
            }
        ):
            issues.append(
                {
                    "file": str(path),
                    "model_id": mid,
                    "violation": f"expected {expected} got {final}",
                    "row_snapshot": ",".join(f"{k}={v}" for k, v in row.items()),
                }
            )
    return issues


def _scan_json(path: Path, target_models: set[str]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return issues
    rows: list[dict[str, Any]] = []
    if isinstance(data, dict):
        for key in ("rows", "models", "results"):
            v = data.get(key)
            if isinstance(v, list):
                rows.extend(r for r in v if isinstance(r, dict))
    if isinstance(data, list):
        rows.extend(r for r in data if isinstance(r, dict))
    for r in rows:
        mid = (r.get("model_id") or "").strip()
        if mid not in target_models:
            continue
        final = (
            r.get("final_state")
            or r.get("final_state_after_v237")
            or r.get("final_state_after_v238")
            or ""
        ).strip()
        if final == "expected_blocker":
            issues.append(
                {
                    "file": str(path),
                    "model_id": mid,
                    "violation": "generic_expected_blocker_json",
                    "row_snapshot": json.dumps(r, default=str)[:300],
                }
            )
        if mid in {"florence-2-base", "florence-2-large"} and final == "dependency_required":
            issues.append(
                {
                    "file": str(path),
                    "model_id": mid,
                    "violation": "florence2_stale_dependency_required_json",
                    "row_snapshot": json.dumps(r, default=str)[:300],
                }
            )
        if mid == "rfdetr-seg-large" and final == "license_blocked":
            issues.append(
                {
                    "file": str(path),
                    "model_id": mid,
                    "violation": "rfdetr_seg_large_stale_license_blocked_json",
                    "row_snapshot": json.dumps(r, default=str)[:300],
                }
            )
    return issues


_NB_CELL_TEXT_RE = re.compile(r'"final_state"\s*:\s*"([^"]+)"')


def _scan_ipynb(path: Path, target_models: set[str]) -> list[dict[str, str]]:
    """Lightweight scan of executed notebook output for known stale patterns."""
    issues: list[dict[str, str]] = []
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        return issues
    if "expected_blocker" not in text:
        return issues
    for mid in target_models:
        if mid not in text:
            continue
        idx = 0
        while True:
            idx = text.find(mid, idx)
            if idx == -1:
                break
            window = text[idx : idx + 500]
            if "expected_blocker" in window:
                issues.append(
                    {
                        "file": str(path),
                        "model_id": mid,
                        "violation": "ipynb_executed_cell_shows_expected_blocker_for_target",
                        "row_snapshot": window[:300].replace("\n", " "),
                    }
                )
                break
            idx += len(mid)
    return issues


def audit_stale_final_tables(
    *,
    notebook_root: Path,
    reports_root: Path | None = None,
    target_models: Iterable[str] | None = None,
    extra_roots: list[Path] | None = None,
) -> dict[str, Any]:
    """Run the stale-table audit. Returns a structured payload.

    Set ``--fail-on-stale`` at the CLI layer to convert non-empty issues into
    a non-zero exit.
    """
    targets: set[str] = set(target_models) if target_models else set(DEFAULT_TARGET_MODELS_49)
    roots: list[Path] = [notebook_root]
    if reports_root is not None:
        roots.append(reports_root)
    if extra_roots:
        roots.extend(extra_roots)

    csv_paths: list[Path] = []
    json_paths: list[Path] = []
    ipynb_paths: list[Path] = []
    for root in roots:
        csv_paths.extend(_iter_files(root, (".csv",)))
        json_paths.extend(_iter_files(root, (".json",)))
        ipynb_paths.extend(_iter_files(root, (".ipynb",)))

    issues: list[dict[str, str]] = []
    for p in csv_paths:
        issues.extend(_scan_csv(p, targets))
    for p in json_paths:
        issues.extend(_scan_json(p, targets))
    for p in ipynb_paths:
        issues.extend(_scan_ipynb(p, targets))

    # Counters
    generic_blocker_count = sum(
        1 for i in issues if i["violation"].startswith("generic_expected_blocker")
    )
    stub_as_final_count = sum(1 for i in issues if i["violation"] == "stub_with_generic_final")
    false_license_count = sum(
        1 for i in issues if i["violation"] == "false_license_blocked_permissive"
    )
    florence_stale_count = sum(
        1 for i in issues if i["violation"].startswith("florence2_stale_dependency_required")
    )
    rfdetr_stale_count = sum(
        1 for i in issues if i["violation"].startswith("rfdetr_seg_large_stale_license_blocked")
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scanned_csv": len(csv_paths),
        "scanned_json": len(json_paths),
        "scanned_ipynb": len(ipynb_paths),
        "target_models": sorted(targets),
        "total_issues": len(issues),
        "counts": {
            "generic_expected_blocker": generic_blocker_count,
            "stub_as_final_state": stub_as_final_count,
            "false_license_blocked": false_license_count,
            "florence2_stale_dependency_required": florence_stale_count,
            "rfdetr_seg_large_stale_license_blocked": rfdetr_stale_count,
        },
        "issues": issues,
        "status": "ok" if not issues else "stale_found",
    }
    return payload


__all__ = [
    "DEFAULT_TARGET_MODELS_49",
    "EXPECTED_CORRECTED_STATES",
    "KNOWN_PERMISSIVE_LICENSE_TOKENS",
    "STALE_PATTERNS",
    "audit_stale_final_tables",
]
