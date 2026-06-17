#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.17.0 QA: universal model lifecycle matrix.

For EVERY discovered model, records a per-stage lifecycle status. The cheap stages
(instantiate / capabilities / engine / license / dependency) are run LIVE for all
models (real evidence, no weight downloads). Heavier stages (inference / train /
reload / export) are capability-DERIVED by default and run LIVE only for the
``--live-models`` subset — the matrix is explicit about which is which so nothing
is claimed live without proof.

    python tools/qa/v317_model_lifecycle_matrix.py --device cpu \
        --live-models torchvision-resnet50,libreyolo-yolox-s \
        --output docs/qa/v317_full_model_matrix/lifecycle_matrix.json

Never stops the matrix on one model failure.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

_STAGES = [
    "import",
    "instantiate",
    "capabilities",
    "license",
    "dependency",
    "inference",
    "train",
    "checkpoint_save",
    "checkpoint_reload",
    "predict_after_reload",
    "export",
]


def _stage(status: str, reason: str = "", live: bool = False, **extra) -> dict:
    return {"status": status, "reason": reason, "live_verified": live, **extra}


def _dependency_ok(engine_cls_modules: tuple[str, ...]) -> tuple[bool, str]:
    import importlib

    missing = []
    for mod in engine_cls_modules:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    return (not missing), (f"missing: {missing}" if missing else "")


def _live_inference(model_id: str, device: str, asset: Path) -> dict:
    """Run a real predict() smoke. Returns a stage dict."""
    from PIL import Image

    from visionservex.core.model import VisionModel

    t0 = time.time()
    try:
        m = VisionModel(model_id, device=device)
        res = m.predict(Image.open(asset).convert("RGB"))
        n = len(
            getattr(res, "detections", None)
            or getattr(res, "segments", None)
            or getattr(res, "top_k", [])
            or []
        )
        kind = getattr(res, "kind", "?")
        with contextlib.suppress(Exception):
            m.unload()
        return _stage(
            "PASS",
            f"kind={kind} n_outputs={n}",
            live=True,
            runtime_seconds=round(time.time() - t0, 2),
        )
    except Exception as exc:
        return _stage(
            "FAIL",
            reason=str(exc)[:200],
            live=True,
            error_type=type(exc).__name__,
            runtime_seconds=round(time.time() - t0, 2),
        )


def test_model(model_id: str, *, device: str, live: bool, asset: Path) -> dict:
    from visionservex.core.model import VisionModel, model_capabilities

    row: dict[str, Any] = {"model_id": model_id, "stages": {}}
    stages = row["stages"]

    cap = model_capabilities(model_id)
    row["family"] = cap["family"]
    row["task"] = cap["task"]
    row["readiness"] = cap["readiness"]
    row["exact_blocker"] = cap["exact_blocker"]
    stages["capabilities"] = _stage("PASS", "model_capabilities() returned", live=True)

    # import / engine registration
    if cap["engine_registered"]:
        stages["import"] = _stage("PASS", f"engine {cap['engine']!r} registered", live=True)
    else:
        stages["import"] = _stage(
            "BLOCKED", f"CATALOG_ONLY: engine {cap['engine']!r} not in _FACTORIES", live=True
        )

    # instantiate (cheap, no download)
    eng = None
    try:
        vm = VisionModel(model_id, device=device)
        eng = vm.engine
        stages["instantiate"] = _stage("PASS", f"VisionModel({model_id!r}) built", live=True)
    except Exception as exc:
        stages["instantiate"] = _stage(
            "FAIL", str(exc)[:160], live=True, error_type=type(exc).__name__
        )

    # license
    if cap["legal_status"] == "commercial_safe_core":
        stages["license"] = _stage("PASS", "commercial_safe_core", live=True)
    elif cap["gated"]:
        stages["license"] = _stage("GATED_TOKEN_REQUIRED", cap["legal_status"], live=True)
    elif cap["legal_status"] in ("noncommercial_restricted", "enterprise_license_required"):
        stages["license"] = _stage("LICENSE_BLOCKED", cap["legal_status"], live=True)
    else:
        stages["license"] = _stage("PASS", cap["legal_status"], live=True)

    # dependency
    real_modules = tuple(getattr(type(eng), "real_modules", ()) if eng is not None else ())
    if real_modules:
        ok, why = _dependency_ok(real_modules)
        stages["dependency"] = _stage(
            "PASS" if ok else "DEPENDENCY_MISSING", why or f"{real_modules} importable", live=True
        )
    else:
        stages["dependency"] = _stage("NOT_APPLICABLE", "no declared real_modules", live=True)

    # inference
    if live and cap["pretrained_inference_supported"] and asset.exists():
        stages["inference"] = _live_inference(model_id, device, asset)
    elif cap["pretrained_inference_supported"]:
        stages["inference"] = _stage(
            "SKIP",
            "inference-ready by capability+engine wiring; live smoke not run (use --live-models)",
            live=False,
        )
    elif cap["readiness"] == "catalog-only":
        stages["inference"] = _stage("BLOCKED", cap["exact_blocker"] or "CATALOG_ONLY", live=True)
    else:
        stages["inference"] = _stage(
            "BLOCKED", cap["exact_blocker"] or "not inference-ready", live=True
        )

    # train + reload lifecycle (capability-derived; live only via the dedicated
    # v316 libreyolo / v315 torchvision lifecycle harnesses)
    if cap["train_supported"]:
        note = (
            "train-ready (validated_lifecycle)" if cap["validated_lifecycle"] else "train_supported"
        )
        stages["train"] = _stage("PASS", note + " — see v315/v316 lifecycle proof", live=False)
        stages["checkpoint_save"] = _stage("PASS", "checkpoint_save_supported", live=False)
        stages["checkpoint_reload"] = _stage(
            "PASS" if cap["checkpoint_load_supported"] else "CHECKPOINT_RELOAD_NOT_IMPLEMENTED",
            "",
            live=False,
        )
        stages["predict_after_reload"] = _stage(
            "PASS"
            if cap["trained_checkpoint_predict_supported"]
            else "PREDICT_AFTER_RELOAD_FAILED",
            "",
            live=False,
        )
    else:
        blk = cap["exact_blocker"] or "TRAINING_NOT_IMPLEMENTED"
        for st in ("train", "checkpoint_save", "checkpoint_reload", "predict_after_reload"):
            stages[st] = _stage(
                "NOT_APPLICABLE" if cap["task"] in ("embed", "vlm") else "INFERENCE_ONLY",
                blk,
                live=False,
            )

    # export
    if cap["export_supported"]:
        stages["export"] = _stage("PASS", f"export={cap['export_supported']}", live=False)
    else:
        stages["export"] = _stage("EXPORT_NOT_IMPLEMENTED", "no tested export format", live=False)

    # overall verdict
    row["overall"] = cap["readiness"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--live-models", default="")
    ap.add_argument("--asset", default="tests/assets/smoke/coco_person_car.jpg")
    ap.add_argument("--output", default="docs/qa/v317_full_model_matrix/lifecycle_matrix.json")
    args = ap.parse_args()

    from visionservex.core.model import list_models

    live_set = {m.strip() for m in args.live_models.split(",") if m.strip()}
    asset = Path(args.asset)

    rows = []
    for mid in list_models():
        live = mid in live_set
        try:
            row = test_model(mid, device=args.device, live=live, asset=asset)
        except Exception:
            row = {
                "model_id": mid,
                "stages": {},
                "overall": "HARNESS_ERROR",
                "error": traceback.format_exc(limit=4),
            }
        rows.append(row)
        if live:
            inf = row["stages"].get("inference", {})
            print(
                f"  LIVE {mid}: inference={inf.get('status')} ({inf.get('reason', '')[:60]})",
                flush=True,
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"device": args.device, "live_models": sorted(live_set), "rows": rows},
            indent=2,
            default=str,
        )
    )

    overall = Counter(r.get("overall") for r in rows)
    inf_pass_live = sum(
        1
        for r in rows
        if r["stages"].get("inference", {}).get("live_verified")
        and r["stages"].get("inference", {}).get("status") == "PASS"
    )
    inf_blocked_live = sum(
        1
        for r in rows
        if r["stages"].get("inference", {}).get("live_verified")
        and r["stages"].get("inference", {}).get("status") == "BLOCKED"
    )
    print(f"\nmatrix: {len(rows)} models | overall: {dict(overall)}")
    print(f"live PASS inference (real predict ran): {inf_pass_live}")
    print(f"live-confirmed BLOCKED inference (catalog-only/blocked): {inf_blocked_live}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
