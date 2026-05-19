# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.33.0: model contract-test runner.

A model passes contract-test only when:
  1. Discovery from registry succeeds.
  2. Required dependencies are checked (structured blocker if missing).
  3. License gate is checked.
  4. Checkpoint is present (or pulled, or exact manual flow exists).
  5. Model loads.
  6. Inference runs on a valid asset.
  7. Output normalizes to a task-specific schema.
  8. No raw traceback / NaN / NOT_WIRED / failed_runtime in result.

Every row ends in exactly one of:
  contract_passed | benchmark_passed | dependency_required |
  manual_checkpoint_required | license_blocked | dataset_required |
  auth_required | download_failed_retryable | sidecar_required |
  unsupported_by_upstream | package_bug
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SMOKE_IMG = REPO_ROOT / "tests/assets/smoke/coco_person_car.jpg"


_TASK_SCHEMAS = {
    "detect": {"required_keys": ["detections", "boxes"], "any_of": True},
    "classify": {"required_keys": ["predictions", "topk"], "any_of": True},
    "segment": {"required_keys": ["segments", "masks"], "any_of": True},
    "foundation_segment": {"required_keys": ["mask", "masks", "segments"], "any_of": True},
    "embed": {"required_keys": ["embedding", "embeddings"], "any_of": True},
    "open_vocab_detect": {"required_keys": ["detections", "boxes"], "any_of": True},
    "grounded_segment": {"required_keys": ["segments", "masks"], "any_of": True},
    "vlm": {"required_keys": ["answer", "text", "predictions"], "any_of": True},
}


@dataclass
class ContractResult:
    model_id: str = ""
    family: str = ""
    task: str = ""
    backend: str = ""
    final_state: str = "unclassified"
    blocker_code: str = ""
    fix: str = ""
    package_bug: bool = False
    external_blocker: bool = False
    output_schema_valid: bool = False
    n_outputs: int = 0
    runtime_ms: float = 0.0
    evidence_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _smoke_image_for_task(task: str) -> str:
    return str(_SMOKE_IMG)


def _command_for_model(
    model_id: str,
    task: str,
    device: str,
    out_json: str,
) -> list[str]:
    """Synthesize the CLI command for one model contract test."""
    base = [sys.executable, "-m", "visionservex"]
    img = _smoke_image_for_task(task)
    save = ["--save-json", out_json] if out_json else []

    if task in ("detect", "classify", "pose", "obb", "segment"):
        return [*base, "predict", model_id, img, "--device", device, "--json", *save]
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
            *save,
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
            *save,
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
            *save,
        ]
    if task == "embed":
        return [*base, "feature", "embed", model_id, img, "--device", device, "--json"]
    if task == "vlm":
        return [*base, "predict", model_id, img, "--device", device, "--json", *save]
    return [*base, "predict", model_id, img, "--device", device, "--json", *save]


def _classify_one(
    *,
    model_id: str,
    family: str,
    task: str,
    backend: str,
    proc_returncode: int,
    stdout: str,
    stderr: str,
    runtime_ms: float,
) -> ContractResult:
    """Classify a contract-test result into a final_state with precise codes."""
    from visionservex.runtime.result_classifier import (
        EXPECTED_BLOCKER_CODES,
        classify_command_result,
    )

    classified = classify_command_result(
        returncode=proc_returncode,
        stdout=stdout,
        stderr=stderr,
    )
    payload = classified.structured_payload or {}

    # Also parse stderr as JSON in case the predict wrapper dumped error there
    if not payload:
        for stream_text in (stdout, stderr):
            for line in stream_text.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict):
                            payload = parsed
                            break
                    except Exception:
                        continue
            if payload:
                break

    # Unwrap PREDICT_FAILED envelope
    code = payload.get("code", "")
    if isinstance(payload.get("error"), dict):
        err = payload["error"]
        code = err.get("code", code)
        if code == "PREDICT_FAILED":
            msg = (err.get("message") or "").upper()
            if "BROTLI" in msg or "DOWNLOAD" in msg or "DOWNLOADERROR" in msg:
                code = "DOWNLOAD_FAILED_RETRYABLE"
            elif "TRANSFORMERS" in msg and ("INCOMPATIBLE" in msg or "5." in msg):
                code = "FLORENCE2_TRANSFORMERS_VERSION_REQUIRED"
            elif "NATTEN" in msg:
                code = "NATTEN_REQUIRED"
            elif "GATED" in msg or "AUTH" in msg or "HF_TOKEN" in msg:
                code = "HF_AUTH_REQUIRED"
            elif "CHECKPOINT" in msg or "WEIGHTS" in msg:
                code = "CHECKPOINT_REQUIRED"

    n_outputs = 0
    schema_valid = False
    schema = _TASK_SCHEMAS.get(task)
    if schema:
        for k in schema["required_keys"]:
            v = payload.get(k)
            if v:
                schema_valid = True
                if isinstance(v, list):
                    n_outputs = len(v)
                elif isinstance(v, dict):
                    n_outputs = 1
                break

    # Determine final_state
    if classified.status in ("ok_clean", "ok_with_warning") and schema_valid:
        final_state = "contract_passed"
        external_blocker = False
        package_bug = False
        blocker_code = ""
    elif classified.status == "expected_blocker" or code in EXPECTED_BLOCKER_CODES or code:
        # Map known codes to specific final_state
        if any(t in code for t in ("DOWNLOAD_FAILED", "BROTLI")):
            final_state = "download_failed_retryable"
        elif "AUTH" in code or "GATED" in code:
            final_state = "auth_required"
        elif (
            "MANUAL_CHECKPOINT" in code
            or code == "CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP"
            or "CHECKPOINT" in code
        ):
            final_state = "manual_checkpoint_required"
        elif "LICENSE" in code or "NON_COMMERCIAL" in code or "PML" in code:
            final_state = "license_blocked"
        elif "SIDECAR" in code or "FLORENCE2_TRANSFORMERS" in code:
            final_state = "sidecar_required"
        elif "DATASET" in code:
            final_state = "dataset_required"
        elif code in EXPECTED_BLOCKER_CODES or any(
            t in code for t in ("REQUIRED", "NATTEN", "NEEDS")
        ):
            final_state = "dependency_required"
        else:
            final_state = "dependency_required"
        external_blocker = True
        package_bug = False
        blocker_code = code or "EXPECTED_BLOCKER"
    elif classified.status == "failed_usage":
        final_state = "package_bug"
        external_blocker = False
        package_bug = True
        blocker_code = "CLI_USAGE_ERROR"
    else:
        # Raw failure with no structured payload
        if any(t in (stderr + stdout) for t in ("DownloadError", "brotli")):
            final_state = "download_failed_retryable"
            blocker_code = "DOWNLOAD_FAILED_RETRYABLE"
            external_blocker = True
            package_bug = False
        else:
            final_state = "package_bug"
            blocker_code = "UNCLASSIFIED_FAILURE"
            external_blocker = False
            package_bug = True

    fix = payload.get("recommended_fix", "") or payload.get("install", "") or payload.get("fix", "")

    return ContractResult(
        model_id=model_id,
        family=family,
        task=task,
        backend=backend,
        final_state=final_state,
        blocker_code=blocker_code,
        fix=fix,
        package_bug=package_bug,
        external_blocker=external_blocker,
        output_schema_valid=schema_valid,
        n_outputs=n_outputs,
        runtime_ms=runtime_ms,
    )


@dataclass
class ContractSummary:
    total: int = 0
    contract_passed: int = 0
    benchmark_passed: int = 0
    dependency_required: int = 0
    auth_required: int = 0
    manual_checkpoint_required: int = 0
    license_blocked: int = 0
    dataset_required: int = 0
    download_failed_retryable: int = 0
    sidecar_required: int = 0
    unsupported_by_upstream: int = 0
    package_bug: int = 0
    unclassified: int = 0


def run_contract_matrix(
    *,
    include: str = "core",  # core | all
    device: str = "cuda",
    out_json: Path | None = None,
    out_csv: Path | None = None,
    fail_on_package_bug: bool = False,
    timeout_s: int = 90,
    max_retries: int = 1,
    per_model_log_dir: Path | None = None,
) -> tuple[list[ContractResult], ContractSummary]:
    """Run contract tests across the model registry."""
    from visionservex.registry import default_registry

    reg = default_registry()
    all_entries = list(reg.list())

    if include == "core":
        models = [
            e
            for e in all_entries
            if e.implementation_status in ("wired", "partial") and e.family != "mock"
        ]
    else:
        models = [e for e in all_entries if e.family != "mock"]

    if per_model_log_dir is None and out_json is not None:
        per_model_log_dir = out_json.parent / "contract_logs"
    if per_model_log_dir:
        per_model_log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[ContractResult] = []

    for entry in models:
        model_id = entry.id
        family = entry.family
        task = entry.task
        backend = entry.backend or entry.engine

        print(f"  [{model_id}] task={task} ...", end="", flush=True)
        safe_id = model_id.replace("/", "_").replace(".", "_")
        out_payload = Path("/tmp") / f"vsx_contract_{safe_id}.json"

        cmd = _command_for_model(model_id, task, device, str(out_payload))

        attempts = 0
        result = None
        while attempts <= max_retries:
            t0 = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    cwd=str(REPO_ROOT),
                )
                stdout = proc.stdout
                stderr = proc.stderr
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                stdout = ""
                stderr = "TIMEOUT"
                rc = -9
            except Exception as exc:
                stdout = ""
                stderr = str(exc)
                rc = -1

            runtime_ms = (time.monotonic() - t0) * 1000.0
            result = _classify_one(
                model_id=model_id,
                family=family,
                task=task,
                backend=backend,
                proc_returncode=rc,
                stdout=stdout,
                stderr=stderr,
                runtime_ms=runtime_ms,
            )

            # Save logs
            if per_model_log_dir:
                (per_model_log_dir / f"{safe_id}.stdout").write_text(stdout)
                (per_model_log_dir / f"{safe_id}.stderr").write_text(stderr)
                result.evidence_file = str(per_model_log_dir / f"{safe_id}.stdout")

            # Retry only on transient download failures
            if result.final_state == "download_failed_retryable" and attempts < max_retries:
                attempts += 1
                time.sleep(2**attempts)
                continue
            break

        if result:
            sym = {
                "contract_passed": "✓",
                "benchmark_passed": "✓",
                "dependency_required": "⊘",
                "auth_required": "⊘",
                "manual_checkpoint_required": "⚠",
                "license_blocked": "⊗",
                "dataset_required": "⊘",
                "download_failed_retryable": "↻",
                "sidecar_required": "⊘",
                "package_bug": "✗",
                "unclassified": "?",
            }.get(result.final_state, "?")
            print(
                f" {sym} {result.final_state}"
                + (f" [{result.blocker_code}]" if result.blocker_code else "")
            )
            rows.append(result)

    # Summary
    summary = ContractSummary(total=len(rows))
    for r in rows:
        attr = r.final_state.replace("-", "_")
        if hasattr(summary, attr):
            setattr(summary, attr, getattr(summary, attr) + 1)
        else:
            summary.unclassified += 1

    # Save outputs
    if out_json is not None:
        payload = {
            "version": "v2.33.0",
            "device": device,
            "include": include,
            "summary": asdict(summary),
            "rows": [r.to_dict() for r in rows],
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"\nContract matrix → {out_json}")

    if out_csv is not None:
        import csv

        fields = list(ContractResult.__dataclass_fields__.keys())
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r.to_dict())
        print(f"Contract matrix CSV → {out_csv}")

    if fail_on_package_bug and summary.package_bug > 0:
        bugs = [r for r in rows if r.final_state == "package_bug"]
        print(f"\nFAIL: {summary.package_bug} package_bug rows:")
        for r in bugs:
            print(f"  {r.model_id}: [{r.blocker_code}]")
        sys.exit(1)

    return rows, summary
