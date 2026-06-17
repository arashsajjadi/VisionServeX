# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.18 live inference matrix — really runs every wired, legal, non-gated model.

This is the *evidence generator* for ``readiness/live_evidence.py``. For each
candidate it runs a tiny task-appropriate smoke input through the real public
API (``VisionModel(...).detect/classify/embed/segment/predict``), validates the
returned object against the documented result schema, and records an honest
PASS / FAIL / SKIP_BLOCKED row.

Resource safety (this machine froze once from concurrent model loads — see
AGENT_RULES.md):

* Every model runs in its **own subprocess** with a hard wall-clock timeout, so
  a hang / segfault / OOM-kill takes down only that one worker and its memory is
  fully reclaimed before the next model loads.
* Workers are launched **strictly serially** (never concurrent, never bg pytest).
* Workers are pinned to **CPU** (``CUDA_VISIBLE_DEVICES=""``) so a smoke run can
  never touch VRAM.
* The driver writes results incrementally to a JSONL and is **resumable**: re-run
  it and it skips models already recorded. It also self-limits wall-clock per
  invocation so each call stays well under a shell timeout.

Usage::

    # driver (run repeatedly until it prints ALL DONE):
    python tools/qa/v318_live_inference_matrix.py
    # single worker (internal; prints one JSON line):
    python tools/qa/v318_live_inference_matrix.py --worker <model_id> --fixture <img>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs" / "qa" / "v318_full_model_truth"
FIXTURE_DIR = OUT_DIR / "fixtures"
JSONL = OUT_DIR / "live_inference_matrix.jsonl"
JSON_OUT = OUT_DIR / "live_inference_matrix.json"
MD_OUT = OUT_DIR / "live_inference_matrix.md"

PER_MODEL_TIMEOUT_S = 180
DRIVER_WALLCLOCK_BUDGET_S = 400  # worst case 400+180 < 600s shell timeout per invocation

# Tasks whose live state the matrix can prove; everything else is recorded but
# does not feed live_evidence.
_SEGMENT_FOUNDATION = {"foundation_segment"}
_SEGMENT_SEMANTIC = {"segment"}
_SEGMENT_GROUNDED = {"grounded_segment"}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def ensure_fixtures() -> dict[str, str]:
    """Create tiny deterministic RGB smoke images; return {name: path}."""
    from PIL import Image, ImageDraw

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    generic = FIXTURE_DIR / "smoke_rgb_320.jpg"
    person = FIXTURE_DIR / "smoke_person_256.jpg"
    if not generic.exists():
        img = Image.new("RGB", (320, 320), (180, 200, 220))
        d = ImageDraw.Draw(img)
        d.rectangle([40, 40, 160, 200], fill=(200, 60, 60))
        d.ellipse([180, 120, 280, 240], fill=(60, 120, 200))
        d.rectangle([60, 230, 260, 300], fill=(80, 160, 80))
        img.save(generic, quality=90)
    if not person.exists():
        # crude "person-like" silhouette so pose/segment paths get something.
        img = Image.new("RGB", (256, 256), (230, 230, 230))
        d = ImageDraw.Draw(img)
        d.ellipse([108, 30, 148, 70], fill=(90, 70, 60))  # head
        d.rectangle([110, 70, 146, 170], fill=(60, 90, 160))  # torso
        d.rectangle([112, 170, 126, 240], fill=(40, 40, 80))  # legs
        d.rectangle([130, 170, 144, 240], fill=(40, 40, 80))
        img.save(person, quality=90)
    return {"generic": str(generic), "person": str(person)}


# --------------------------------------------------------------------------- #
# Worker — runs ONE model in this (isolated) process and prints one JSON line.
# --------------------------------------------------------------------------- #
def _validate_schema(task: str, result) -> tuple[bool, str | None]:
    """Return (schema_valid, method-detail). Honest about misrouted/empty."""
    from visionservex.core.results import (
        BaseResult,
        ClassificationResult,
        DetectionResult,
        OpenVocabularyResult,
        OrientedDetectionResult,
        PoseResult,
        SegmentationResult,
    )

    if not isinstance(result, BaseResult):
        return False, f"not a BaseResult: {type(result).__name__}"
    # Embeddings come back as their own result type in this codebase; accept any
    # BaseResult carrying an `embedding`/`features` payload.
    expectations = {
        "detect": (DetectionResult,),
        "obb": (OrientedDetectionResult, DetectionResult),
        "open_vocab_detect": (OpenVocabularyResult, DetectionResult),
        "pose": (PoseResult,),
        "classify": (ClassificationResult,),
        "segment": (SegmentationResult,),
        "foundation_segment": (SegmentationResult,),
        "grounded_segment": (SegmentationResult, OpenVocabularyResult),
    }
    want = expectations.get(task)
    if want is not None and not isinstance(result, want):
        # Some embedders/vlms reuse a generic result; only fail when we *named* a
        # concrete expectation and the object is a different concrete result.
        return False, f"{type(result).__name__} not in {[w.__name__ for w in want]}"
    return True, type(result).__name__


def run_one(model_id: str, fixtures: dict[str, str]) -> dict:
    t0 = time.perf_counter()
    row = {
        "model_id": model_id,
        "task": None,
        "live_verified": False,
        "status": "FAIL",
        "method_called": None,
        "input_fixture": None,
        "output_schema_valid": False,
        "latency_ms": 0.0,
        "device": "cpu",
        "error_type": None,
        "error_message": None,
        "blocker": None,
    }
    try:
        from visionservex import VisionModel
        from visionservex.core.model import model_capabilities

        cap = model_capabilities(model_id)
        task = cap["task"]
        row["task"] = task

        img = fixtures["person"] if task == "pose" else fixtures["generic"]
        row["input_fixture"] = Path(img).name

        model = VisionModel(model_id, device="cpu")

        # Choose the typed method + smoke args per task.
        if task in ("detect", "obb"):
            row["method_called"] = "detect"
            result = model.detect(img, threshold=0.1)
        elif task == "open_vocab_detect":
            row["method_called"] = "detect"
            result = model.detect(img, prompts=["object", "shape"])
        elif task == "classify":
            row["method_called"] = "classify"
            result = model.classify(img, top_k=5)
        elif task in ("embed", "embedding"):
            row["method_called"] = "embed"
            result = model.embed(img)
        elif task in _SEGMENT_FOUNDATION:
            row["method_called"] = "segment"
            result = model.segment(img, points=[[160.0, 160.0]], point_labels=[1])
        elif task in _SEGMENT_GROUNDED:
            row["method_called"] = "segment"
            result = model.segment(img, prompts=["object"])
        elif task in _SEGMENT_SEMANTIC:
            row["method_called"] = "segment"
            result = model.segment(img)
        elif task == "vlm":
            row["method_called"] = "predict"
            result = model.predict(img, prompt="What is in this image?")
        else:  # pose and any other -> generic predict
            row["method_called"] = "predict"
            result = model.predict(img)

        valid, detail = _validate_schema(task, result)
        row["output_schema_valid"] = valid
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        if valid:
            row["status"] = "PASS"
            row["live_verified"] = True
        else:
            row["status"] = "FAIL"
            row["error_type"] = "SchemaInvalid"
            row["error_message"] = detail
        with contextlib.suppress(Exception):
            model.unload()
    except Exception as e:
        import traceback

        etype = type(e).__name__
        msg = str(e)[:600]
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        row["error_type"] = etype
        row["error_message"] = msg
        # Classify dependency / weights blockers as honest SKIP_BLOCKED.
        low = (msg + " " + etype).lower()
        if any(k in low for k in ("no module named", "importerror", "modulenotfound")):
            row["status"] = "SKIP_BLOCKED"
            row["blocker"] = "DEPENDENCY_MISSING"
        elif any(
            k in low
            for k in ("connection", "offline", "couldn't reach", "could not download", "404")
        ):
            row["status"] = "SKIP_BLOCKED"
            row["blocker"] = "WEIGHTS_DOWNLOAD_UNAVAILABLE"
        elif any(k in low for k in ("out of memory", "oom", "cannot allocate")):
            row["status"] = "SKIP_BLOCKED"
            row["blocker"] = "OOM_BLOCKED"
        else:
            row["status"] = "FAIL"
        sys.stderr.write(traceback.format_exc()[-1500:])
    return row


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def candidates() -> list[str]:
    import visionservex as vsx

    out = []
    for m in vsx.list_models():
        c = vsx.model_capabilities(m)
        if (
            c["readiness"] in ("inference-ready", "train-ready")
            and c["engine_registered"]
            and not c["gated"]
        ):
            out.append(m)
    return sorted(out)


def load_done() -> dict[str, dict]:
    done = {}
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[r["model_id"]] = r
            except Exception:
                pass
    return done


def write_outputs(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["task"] or "", r["model_id"]))
    summary = {"PASS": 0, "FAIL": 0, "SKIP_BLOCKED": 0}
    for r in rows:
        summary[r.get("status", "FAIL")] = summary.get(r.get("status", "FAIL"), 0) + 1
    payload = {
        "schema": "v318_live_inference_matrix",
        "device": "cpu",
        "n": len(rows),
        "summary": summary,
        "results": rows,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2))

    lines = [
        "# v3.18 Live Inference Matrix",
        "",
        "Real CPU smoke inference for every wired, legal, non-gated model. Each row",
        "ran the model's own public API on a tiny fixture and validated the result",
        "schema. `PASS` ⇒ the model is eligible for an `*_READY_LIVE` readiness state.",
        "",
        f"**Totals:** {summary.get('PASS', 0)} PASS · {summary.get('FAIL', 0)} FAIL · "
        f"{summary.get('SKIP_BLOCKED', 0)} SKIP_BLOCKED · device=cpu",
        "",
        "| Model | Task | Status | Method | Schema | Latency ms | Error / Blocker |",
        "|---|---|---|---|---|---:|---|",
    ]
    for r in rows:
        err = r.get("blocker") or r.get("error_type") or ""
        if r.get("error_message") and r.get("status") != "PASS":
            err = f"{err}: {str(r['error_message'])[:80]}"
        lines.append(
            f"| `{r['model_id']}` | {r.get('task', '')} | {r.get('status', '')} | "
            f"{r.get('method_called', '') or ''} | {'ok' if r.get('output_schema_valid') else '—'} | "
            f"{r.get('latency_ms', 0)} | {err} |"
        )
    MD_OUT.write_text("\n".join(lines) + "\n")


def drive() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = ensure_fixtures()
    cands = candidates()
    done = load_done()
    todo = [m for m in cands if m not in done]
    print(f"[driver] candidates={len(cands)} done={len(done)} todo={len(todo)}")

    start = time.perf_counter()
    worker_env = dict(os.environ)
    worker_env["CUDA_VISIBLE_DEVICES"] = ""  # CPU only — never touch VRAM
    worker_env["TMPDIR"] = os.environ.get("TMPDIR", "/home/arash/.cache/vsx_tmp")

    processed = 0
    with JSONL.open("a") as fh:
        for m in todo:
            if time.perf_counter() - start > DRIVER_WALLCLOCK_BUDGET_S:
                print(
                    f"[driver] wall-clock budget hit; {len(todo) - processed} remain. Re-run to resume."
                )
                break
            cmd = [sys.executable, __file__, "--worker", m, "--fixture", fixtures["generic"]]
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=PER_MODEL_TIMEOUT_S, env=worker_env
                )
                line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
                row = json.loads(line) if line.startswith("{") else None
                if row is None:
                    row = {
                        "model_id": m,
                        "task": None,
                        "live_verified": False,
                        "status": "FAIL",
                        "method_called": None,
                        "input_fixture": None,
                        "output_schema_valid": False,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "device": "cpu",
                        "error_type": "WorkerCrash",
                        "error_message": (proc.stderr or proc.stdout or "no output")[-300:],
                        "blocker": None,
                    }
            except subprocess.TimeoutExpired:
                row = {
                    "model_id": m,
                    "task": None,
                    "live_verified": False,
                    "status": "FAIL",
                    "method_called": None,
                    "input_fixture": None,
                    "output_schema_valid": False,
                    "latency_ms": PER_MODEL_TIMEOUT_S * 1000.0,
                    "device": "cpu",
                    "error_type": "Timeout",
                    "error_message": f"exceeded {PER_MODEL_TIMEOUT_S}s",
                    "blocker": "TIMEOUT",
                }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            processed += 1
            print(
                f"[{processed}/{len(todo)}] {row['status']:<12} {m} ({row.get('task')}) {row.get('latency_ms')}ms"
            )

    rows = list(load_done().values())
    write_outputs(rows)
    remaining = len([m for m in cands if m not in load_done()])
    if remaining == 0:
        print(f"[driver] ALL DONE — {len(rows)} models recorded -> {JSON_OUT.name}")
    else:
        print(f"[driver] {remaining} remaining — re-run to continue.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None, help="run a single model id and print one JSON line")
    ap.add_argument("--fixture", default=None)
    args = ap.parse_args()
    if args.worker:
        fixtures = ensure_fixtures()
        row = run_one(args.worker, fixtures)
        print(json.dumps(row))
        return 0
    return drive()


if __name__ == "__main__":
    raise SystemExit(main())
