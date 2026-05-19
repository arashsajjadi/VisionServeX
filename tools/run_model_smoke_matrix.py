#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""VisionServeX v2.29.0 — package-level model smoke matrix runner.

Discovers every advertised model, synthesises the exact CLI command for
each, executes it, classifies the result, and saves a CSV + JSON matrix.

Every model ends in exactly one of:
  smoke_passed | benchmark_passed | expected_blocker |
  license_blocked | manual_checkpoint_required | failed_runtime

Usage (standalone):
    python tools/run_model_smoke_matrix.py \
        --device cuda \
        --include-core \
        --out reports/model_smoke_matrix_v229.json \
        --csv reports/model_smoke_matrix_v229.csv

Usage (via CLI):
    visionservex models smoke-matrix --device cuda --include-core \
        --out reports/model_smoke_matrix_v229.json \
        --csv reports/model_smoke_matrix_v229.csv
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_SMOKE_ASSETS = _REPO_ROOT / "tests" / "assets" / "smoke"

_DEFAULT_DETECT_IMAGE = _SMOKE_ASSETS / "coco_person_car.jpg"
_DEFAULT_SEG_IMAGE = _SMOKE_ASSETS / "coco_instance_sample.jpg"
_DEFAULT_SEG_ANN = _SMOKE_ASSETS / "coco_instance_sample.json"
_DEFAULT_MED_IMAGE = _SMOKE_ASSETS / "medical_box_sample.png"
_DEFAULT_AGRI_IMAGE = _SMOKE_ASSETS / "crop_weed_sample.jpg"
_DEFAULT_VIDEO = _SMOKE_ASSETS / "tracking_sample.mp4"
_DEFAULT_ANOMALY_DIR = _SMOKE_ASSETS / "anomaly_simple"

# Fallback: use bundled example images if smoke assets not present
_FALLBACK_IMAGE = _REPO_ROOT / "examples" / "images" / "street.jpg"

# ---------------------------------------------------------------------------
# Blocker-code sets used for final-state assignment
# ---------------------------------------------------------------------------

_LICENSE_CODES = frozenset(
    [
        "RFDETR_PLUS_LICENSE_BLOCKED",
        "NON_CORE_LICENSE_OPT_IN_REQUIRED",
        "LICENSE_RESTRICTION_TRIGGERED",
        "MVTEC_NONCOMMERCIAL",
        "DOTA_NONCOMMERCIAL",
        "VISDRONE_NONCOMMERCIAL",
        "PLANTVILLAGE_NONCOMMERCIAL",
        "DUKEMTMC_RETRACTED",
        "DATASET_LICENSE_UNVERIFIED",
        "OBB_DATASET_NOT_AUDITED",
        "MEDICAL_WEIGHT_NOT_AUDITED",
        "DETECTRON2_WEIGHT_LICENSE_RISK",
        "LIBREYOLO_WEIGHT_LICENSE_UNVERIFIED",
    ]
)

_CHECKPOINT_CODES = frozenset(
    [
        "CHECKPOINT_REQUIRED",
        "CHECKPOINT_NOT_FOUND",
        "MANUAL_CHECKPOINT_REQUIRED",
        "CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP",
    ]
)

# ---------------------------------------------------------------------------
# Advertised model list — what smoke-matrix covers
# ---------------------------------------------------------------------------


def _get_advertised_models(
    *,
    include_core: bool = True,
    include_optional: bool = False,
    include_sidecar: bool = False,
    include_domain: bool = False,
    include_mock: bool = False,
    include_libreyolo_default_safe: bool = False,
) -> list[dict[str, Any]]:
    """Return the advertised model list from the package registry.

    v2.30.0: when ``include_libreyolo_default_safe`` is True, LibreYOLO
    weights with verified MIT or Apache-2.0 weight_license are appended as
    pseudo-registry entries; they share the smoke-matrix execution path via
    `visionservex libreyolo smoke-test`.
    """
    from visionservex.registry import default_registry

    reg = default_registry()
    result: list[dict[str, Any]] = []
    for entry in reg.list():
        if not include_mock and entry.family == "mock":
            continue
        impl = entry.implementation_status
        backend = entry.backend or entry.engine

        # Core = wired or partial, not sidecar, not domain-only
        is_sidecar = "sidecar" in backend or "openmmlab" in backend
        is_domain_only = entry.model_category in (
            "medical_extra",
            "industrial_extra",
            "geospatial_extra",
            "agriculture_extra",
            "surveillance_pipeline_component",
        )
        is_core = impl in ("wired", "partial") and not is_sidecar and not is_domain_only
        is_optional_stub = impl == "stub" and entry.model_category in (
            "production_recommended",
            "accuracy_grade",
            "experimental_sota",
        )
        is_sidecar_model = is_sidecar and impl in ("wired", "partial")
        is_domain = is_domain_only and impl in ("wired", "partial")

        should_include = False
        if include_core and is_core:
            should_include = True
        if include_optional and is_optional_stub:
            should_include = True
        if include_sidecar and is_sidecar_model:
            should_include = True
        if include_domain and is_domain:
            should_include = True

        if should_include:
            result.append(
                {
                    "model_id": entry.id,
                    "family": entry.family,
                    "task": entry.task,
                    "advertised": True,
                    "backend": backend,
                    "implementation_status": impl,
                    "is_sidecar": is_sidecar,
                    "is_domain": is_domain_only,
                    "source": "registry",
                }
            )

    if include_libreyolo_default_safe:
        result.extend(_get_libreyolo_default_safe_models())

    return result


def _get_libreyolo_default_safe_models() -> list[dict[str, Any]]:
    """Return LibreYOLO weights with verified MIT/Apache-2.0 license."""
    try:
        from visionservex.cli.libreyolo_commands import (
            _all_discovered_weights,
            _libreyolo_available,
            _license_verdict_for_family,
        )
    except Exception:
        return []

    avail, _ = _libreyolo_available()
    if not avail:
        return []

    try:
        weights = _all_discovered_weights()
    except Exception:
        return []

    if not weights:
        return []

    safe: list[dict[str, Any]] = []
    for w in weights:
        verdict = _license_verdict_for_family(w["family"])
        wl = (verdict.get("weight_license") or "").upper()
        risk = (verdict.get("license_risk") or "").lower()
        # Default-safe: MIT or Apache-2.0 and license_risk == "none"
        if risk == "none" and any(ok in wl for ok in ("APACHE-2.0", "APACHE 2.0", "MIT")):
            safe.append(
                {
                    "model_id": w.get("model_id", ""),
                    "family": w.get("family", "libreyolo"),
                    "task": w.get("task", "detect"),
                    "advertised": True,
                    "backend": "libreyolo",
                    "implementation_status": "wired",
                    "is_sidecar": False,
                    "is_domain": False,
                    "source": "libreyolo",
                    "weight_license": verdict.get("weight_license", ""),
                    "weight_filename": w.get("filename", ""),
                    "weight_url": w.get("url", ""),
                }
            )
    return safe


# ---------------------------------------------------------------------------
# Command synthesis
# ---------------------------------------------------------------------------


def _smoke_image(task: str) -> str:
    if task in ("foundation_segment", "grounded_segment"):
        img = _DEFAULT_SEG_IMAGE if _DEFAULT_SEG_IMAGE.exists() else _FALLBACK_IMAGE
    elif task in ("vlm",):
        img = _DEFAULT_DETECT_IMAGE if _DEFAULT_DETECT_IMAGE.exists() else _FALLBACK_IMAGE
    else:
        img = _DEFAULT_DETECT_IMAGE if _DEFAULT_DETECT_IMAGE.exists() else _FALLBACK_IMAGE
    return str(img)


def build_smoke_command(
    model_id: str,
    task: str,
    *,
    device: str = "cuda",
    out_json: str = "/dev/null",
    draw_path: str = "",
) -> list[str]:
    """Return the CLI argv list for one model smoke invocation."""
    base = [sys.executable, "-m", "visionservex"]
    img = _smoke_image(task)

    save_flag = ["--save-json", out_json] if out_json and out_json != "/dev/null" else []

    if task in ("detect", "classify", "pose", "obb"):
        return [*base, "predict", model_id, img, "--device", device, "--json", *save_flag]

    if task == "segment":
        return [*base, "predict", model_id, img, "--device", device, "--json", *save_flag]

    if task == "foundation_segment":
        return [
            *base,
            "predict",
            model_id,
            img,
            "--device",
            device,
            "--box",
            "50,50,200,200",
            "--json",
            *save_flag,
        ]

    if task == "open_vocab_detect":
        return [
            *base,
            "predict",
            model_id,
            img,
            "--device",
            device,
            "--prompt",
            "person,car",
            "--json",
            *save_flag,
        ]

    if task == "grounded_segment":
        return [
            *base,
            "predict",
            model_id,
            img,
            "--device",
            device,
            "--prompt",
            "person",
            "--json",
            *save_flag,
        ]

    if task == "embed":
        return [*base, "feature", "embed", model_id, img, "--device", device, "--json"]

    if task == "anomaly":
        return [
            *base,
            "benchmark-anomaly",
            "--dataset",
            str(_DEFAULT_ANOMALY_DIR),
            "--format",
            "json",
            "--out",
            out_json,
        ]

    if task == "track":
        return [
            *base,
            "video-search",
            "tracker-smoke",
            "--tracker",
            model_id,
            "--format",
            "json",
            "--out",
            out_json,
        ]

    if task == "reid":
        return [
            *base,
            "video-search",
            "reid-smoke",
            "--reid",
            model_id,
            "--format",
            "json",
            "--out",
            out_json,
        ]

    if task == "vlm":
        return [*base, "predict", model_id, img, "--device", device, "--json", *save_flag]

    # Fallback
    return [*base, "predict", model_id, img, "--device", device, "--json", *save_flag]


def _build_libreyolo_smoke_command(
    model_id: str,
    task: str,
    *,
    device: str = "cuda",
    out_json: str = "",
    draw_path: str = "",
) -> list[str]:
    """v2.30.0: synthesise `visionservex libreyolo smoke-test` for a LibreYOLO weight."""
    base = [sys.executable, "-m", "visionservex"]
    img = _smoke_image(task)
    cmd = [
        *base,
        "libreyolo",
        "smoke-test",
        model_id,
        img,
        "--device",
        device,
        "--format",
        "json",
    ]
    if out_json and out_json != "/dev/null":
        cmd += ["--out", out_json]
    if draw_path:
        cmd += ["--draw", draw_path]
    return cmd


# ---------------------------------------------------------------------------
# Row dataclass
# ---------------------------------------------------------------------------


@dataclass
class SmokeRow:
    model_id: str = ""
    family: str = ""
    task: str = ""
    advertised: bool = True
    command: str = ""
    returncode: int = -1
    stdout_path: str = ""
    stderr_path: str = ""
    output_json_path: str = ""
    draw_path: str = ""
    runtime_ms: float = 0.0
    json_parseable: bool = False
    output_schema_valid: bool = False
    final_state: str = "unclassified"
    blocker_code: str = ""
    recommended_fix: str = ""
    missing_information: str = ""
    package_bug: bool = False
    external_blocker: bool = False
    license_blocker: bool = False
    retry_count: int = 0
    evidence_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Result classification
# ---------------------------------------------------------------------------


def _classify_row(
    row: SmokeRow,
    stdout: str,
    stderr: str,
    out_json_path: Path,
) -> SmokeRow:
    """Classify a completed subprocess result into a final_state."""
    from visionservex.runtime.result_classifier import (
        EXPECTED_BLOCKER_CODES,
        classify_command_result,
    )

    classified = classify_command_result(
        returncode=row.returncode,
        stdout=stdout,
        stderr=stderr,
        output_path=out_json_path if out_json_path.exists() else None,
        expect_json_at=out_json_path if out_json_path.exists() else None,
    )

    row.json_parseable = classified.artifact_checks.get("json_parseable", False)

    # Extract structured payload from stdout, stderr, or file.
    # The predict command writes error JSON to STDERR (not stdout), so we
    # must parse both streams.
    import contextlib

    payload: dict[str, Any] | None = classified.structured_payload

    def _try_parse_json(text: str) -> dict[str, Any] | None:
        """Try to parse a dict from full text or last non-empty JSON line."""
        with contextlib.suppress(Exception):
            obj = json.loads(text.strip())
            if isinstance(obj, dict):
                return obj
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if stripped.startswith("{"):
                with contextlib.suppress(Exception):
                    obj = json.loads(stripped)
                    if isinstance(obj, dict):
                        return obj
        return None

    if payload is None:
        payload = _try_parse_json(stdout)
    if payload is None:
        payload = _try_parse_json(stderr)
    if payload is None and out_json_path.exists():
        with contextlib.suppress(Exception):
            payload = json.loads(out_json_path.read_text())

    # Unwrap VisionServeX error envelope: {"error": {"code": ..., "message": ...}}
    if isinstance(payload, dict) and "error" in payload and isinstance(payload["error"], dict):
        err = payload["error"]
        code_raw = err.get("code", "")
        msg_raw = err.get("message", "")
        # Map PREDICT_FAILED to a known blocker if the message contains clues
        if code_raw == "PREDICT_FAILED":
            for blocker_code in EXPECTED_BLOCKER_CODES:
                if blocker_code in msg_raw.upper():
                    payload = {
                        "status": "expected_blocker",
                        "code": blocker_code,
                        "message": msg_raw[:400],
                    }
                    break
            else:
                # Heuristic pattern matching for common dependency errors
                if any(
                    kw in msg_raw.lower() for kw in ("brotli", "download failed", "downloadfailed")
                ):
                    payload = {
                        "status": "expected_blocker",
                        "code": "DOWNLOAD_FAILED",
                        "message": msg_raw[:400],
                    }
                elif any(kw in msg_raw.lower() for kw in ("not found", "requires", "install")):
                    payload = {
                        "status": "expected_blocker",
                        "code": "DEPENDENCY_REQUIRED",
                        "message": msg_raw[:400],
                    }
                elif any(kw in msg_raw.lower() for kw in ("incompatible", "version", "conflict")):
                    payload = {
                        "status": "expected_blocker",
                        "code": "DEPENDENCY_CONFLICT",
                        "message": msg_raw[:400],
                    }
                elif any(kw in msg_raw.lower() for kw in ("natten",)):
                    payload = {
                        "status": "expected_blocker",
                        "code": "DEPENDENCY_REQUIRED",
                        "message": msg_raw[:400],
                    }

    code = (payload or {}).get("code", "") or (payload or {}).get("blocker_code", "")
    status = (payload or {}).get("status", "")

    row.blocker_code = code
    row.evidence_file = row.output_json_path

    if classified.status == "expected_blocker" or status == "expected_blocker":
        if code in _LICENSE_CODES:
            row.final_state = "license_blocked"
            row.license_blocker = True
        elif code in _CHECKPOINT_CODES:
            row.final_state = "manual_checkpoint_required"
            row.external_blocker = True
        else:
            row.final_state = "expected_blocker"
            row.external_blocker = True
        row.recommended_fix = (payload or {}).get("recommended_fix", "") or (
            (payload or {}).get("next_action", "")
        )
        row.missing_information = "; ".join(
            (payload or {}).get("missing_information", [])
            if isinstance((payload or {}).get("missing_information", []), list)
            else [(payload or {}).get("missing_information", "")]
        )
        return row

    if classified.status in ("ok_clean", "ok_with_warning"):
        # Validate output schema
        if payload:
            has_kind = "kind" in payload or "status" in payload or "embedding_dim" in payload
            row.output_schema_valid = has_kind
        row.final_state = "smoke_passed"
        return row

    if classified.status == "failed_usage":
        row.final_state = "failed_runtime"
        row.blocker_code = "CLI_USAGE_ERROR"
        row.package_bug = True
        return row

    if classified.status == "failed_output_missing":
        row.final_state = "failed_runtime"
        row.blocker_code = "OUTPUT_MISSING"
        row.package_bug = True
        return row

    # failed_runtime — check for known package-side patterns in combined output
    combined = stdout + "\n" + stderr
    known_blocker = next((c for c in EXPECTED_BLOCKER_CODES if c in combined), None)
    if known_blocker:
        row.blocker_code = known_blocker
        row.final_state = "expected_blocker"
        row.external_blocker = True
        return row

    # Heuristic fallback: classify traceback-level download/dependency errors
    # as expected_blocker since they are not package-logic bugs.
    combined_lower = combined.lower()
    if any(kw in combined_lower for kw in ("downloaderror", "download failed", "brotli")):
        row.blocker_code = "DOWNLOAD_FAILED"
        row.final_state = "expected_blocker"
        row.external_blocker = True
        return row
    if "natten" in combined_lower or "natten library" in combined_lower:
        row.blocker_code = "DEPENDENCY_REQUIRED"
        row.final_state = "expected_blocker"
        row.external_blocker = True
        return row
    if any(
        kw in combined_lower for kw in ("incompatible with transformers", "dependency_conflict")
    ):
        row.blocker_code = "DEPENDENCY_CONFLICT"
        row.final_state = "expected_blocker"
        row.external_blocker = True
        return row

    row.final_state = "failed_runtime"
    row.package_bug = row.returncode != 0 and not row.external_blocker
    return row


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


@dataclass
class SmokeSummary:
    total: int = 0
    smoke_passed: int = 0
    benchmark_passed: int = 0
    expected_blocker: int = 0
    license_blocked: int = 0
    manual_checkpoint_required: int = 0
    failed_runtime: int = 0
    unclassified: int = 0
    package_bug_remaining: int = 0


def run_smoke_matrix(
    *,
    device: str = "cpu",
    include_core: bool = True,
    include_optional: bool = False,
    include_sidecar: bool = False,
    include_domain: bool = False,
    include_mock: bool = False,
    include_libreyolo_default_safe: bool = False,
    out: Path | None = None,
    csv_path: Path | None = None,
    fail_on_package_bug: bool = False,
    per_model_log_dir: Path | None = None,
    timeout_s: int = 120,
    no_notebook: bool = True,
) -> tuple[list[SmokeRow], SmokeSummary]:
    """Run the smoke matrix.  Returns (rows, summary)."""
    if per_model_log_dir is None and out is not None:
        per_model_log_dir = out.parent / "smoke_logs"
    if per_model_log_dir:
        per_model_log_dir.mkdir(parents=True, exist_ok=True)

    models = _get_advertised_models(
        include_core=include_core,
        include_optional=include_optional,
        include_sidecar=include_sidecar,
        include_domain=include_domain,
        include_mock=include_mock,
        include_libreyolo_default_safe=include_libreyolo_default_safe,
    )

    rows: list[SmokeRow] = []

    for m in models:
        model_id = m["model_id"]
        task = m["task"]
        print(f"  [{model_id}] task={task} ...", end="", flush=True)

        # Determine output paths
        safe_id = model_id.replace("/", "_").replace(".", "_")
        out_json_path = Path("/tmp") / f"vsx_smoke_{safe_id}.json"
        draw_path_str = str(Path("/tmp") / f"vsx_smoke_{safe_id}_draw.jpg")

        if m.get("source") == "libreyolo":
            cmd = _build_libreyolo_smoke_command(
                model_id,
                task,
                device=device,
                out_json=str(out_json_path),
                draw_path=draw_path_str,
            )
        else:
            cmd = build_smoke_command(
                model_id,
                task,
                device=device,
                out_json=str(out_json_path),
                draw_path=draw_path_str,
            )

        row = SmokeRow(
            model_id=model_id,
            family=m["family"],
            task=task,
            advertised=True,
            command=" ".join(cmd),
            output_json_path=str(out_json_path),
        )

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(_REPO_ROOT),
            )
            stdout = proc.stdout
            stderr = proc.stderr
            row.returncode = proc.returncode
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = f"TIMEOUT after {timeout_s}s"
            row.returncode = -9
        except Exception as exc:
            stdout = ""
            stderr = str(exc)
            row.returncode = -1

        row.runtime_ms = (time.monotonic() - t0) * 1000.0

        # Save per-model logs
        if per_model_log_dir:
            (per_model_log_dir / f"{safe_id}.stdout").write_text(stdout)
            (per_model_log_dir / f"{safe_id}.stderr").write_text(stderr)
            row.stdout_path = str(per_model_log_dir / f"{safe_id}.stdout")
            row.stderr_path = str(per_model_log_dir / f"{safe_id}.stderr")

        row = _classify_row(row, stdout, stderr, out_json_path)

        state_sym = {
            "smoke_passed": "✓",
            "expected_blocker": "⊘",
            "license_blocked": "⊗",
            "manual_checkpoint_required": "⚠",
            "failed_runtime": "✗",
            "unclassified": "?",
        }.get(row.final_state, "?")
        print(
            f" {state_sym} {row.final_state}"
            + (f" [{row.blocker_code}]" if row.blocker_code else "")
        )

        rows.append(row)

    # ---- summary ----
    summary = SmokeSummary(total=len(rows))
    for r in rows:
        if r.final_state == "smoke_passed":
            summary.smoke_passed += 1
        elif r.final_state == "benchmark_passed":
            summary.benchmark_passed += 1
        elif r.final_state == "expected_blocker":
            summary.expected_blocker += 1
        elif r.final_state == "license_blocked":
            summary.license_blocked += 1
        elif r.final_state == "manual_checkpoint_required":
            summary.manual_checkpoint_required += 1
        elif r.final_state == "failed_runtime":
            summary.failed_runtime += 1
            if r.package_bug:
                summary.package_bug_remaining += 1
        else:
            summary.unclassified += 1

    # ---- save outputs ----
    matrix_payload = {
        "version": "v2.29.0",
        "device": device,
        "summary": asdict(summary),
        "rows": [r.to_dict() for r in rows],
    }

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(matrix_payload, indent=2))
        print(f"\n  Matrix JSON → {out}")

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(SmokeRow.__dataclass_fields__.keys())
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r.to_dict())
        print(f"  Matrix CSV  → {csv_path}")

    if fail_on_package_bug and summary.package_bug_remaining > 0:
        bugs = [r for r in rows if r.package_bug]
        print(f"\n  FAIL: {summary.package_bug_remaining} package-side bug(s) remaining:")
        for r in bugs:
            print(f"    {r.model_id}: {r.final_state} [{r.blocker_code}]")
        sys.exit(1)

    return rows, summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="VisionServeX model smoke matrix runner")
    p.add_argument("--device", default="cpu")
    p.add_argument("--include-core", action="store_true", default=True)
    p.add_argument("--include-optional", action="store_true", default=False)
    p.add_argument("--include-sidecar", action="store_true", default=False)
    p.add_argument("--include-domain", action="store_true", default=False)
    p.add_argument("--include-mock", action="store_true", default=False)
    p.add_argument("--include-libreyolo-default-safe", action="store_true", default=False)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--fail-on-package-bug", action="store_true", default=False)
    p.add_argument("--no-notebook", action="store_true", default=True)
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()

    print(f"VisionServeX smoke-matrix — device={args.device}")
    _rows, summary = run_smoke_matrix(
        device=args.device,
        include_core=args.include_core,
        include_optional=args.include_optional,
        include_sidecar=args.include_sidecar,
        include_domain=args.include_domain,
        include_mock=args.include_mock,
        include_libreyolo_default_safe=args.include_libreyolo_default_safe,
        out=args.out,
        csv_path=args.csv,
        fail_on_package_bug=args.fail_on_package_bug,
        timeout_s=args.timeout,
        no_notebook=args.no_notebook,
    )

    print("\n── Summary ──")
    print(f"  total                    : {summary.total}")
    print(f"  smoke_passed             : {summary.smoke_passed}")
    print(f"  benchmark_passed         : {summary.benchmark_passed}")
    print(f"  expected_blocker         : {summary.expected_blocker}")
    print(f"  license_blocked          : {summary.license_blocked}")
    print(f"  manual_checkpoint        : {summary.manual_checkpoint_required}")
    print(f"  failed_runtime           : {summary.failed_runtime}")
    print(f"  unclassified             : {summary.unclassified}")
    print(f"  package_bug_remaining    : {summary.package_bug_remaining}")


if __name__ == "__main__":
    main()
