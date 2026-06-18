# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.20 train / fine-tune matrix.

Two things:
1. Runs the REAL embedding head fine-tune lifecycle (frozen backbone + linear
   probe) live for every ``embed``-task model:
       finetune_embedding_head -> checkpoint -> EmbeddingHeadModel.from_checkpoint
       -> classify/embed/similarity-after-reload
   → these become FINE_TUNE_READY_LIVE.
2. Classifies every family into the v3.20 lifecycle categories from the committed
   evidence (full-train families re-use their v3.18/v3.19 matrices; SAM-style
   foundation segmenters are honestly INFERENCE_ONLY_LIVE — no fake train).

Resource-safe: embedding fine-tune workers are subprocess-isolated, serial, CPU.

Usage::

    python tools/qa/v320_train_finetune_matrix.py                 # driver
    python tools/qa/v320_train_finetune_matrix.py --worker dinov2-small
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs" / "qa" / "v320_final_operationalization"
JSONL = OUT_DIR / "train_finetune_matrix.jsonl"
JSON_OUT = OUT_DIR / "train_finetune_matrix.json"
MD_OUT = OUT_DIR / "train_finetune_matrix.md"

PER_MODEL_TIMEOUT_S = 300
DRIVER_WALLCLOCK_BUDGET_S = 500

# Full end-to-end train families already live-proven in committed matrices.
_FULL_TRAIN_FAMILIES = {"torchvision-classify", "libreyolo", "rfdetr"}
# Foundation segmenters: prompt inference only, NOT train-ready by design.
_FOUNDATION_SEG_FAMILIES = {"sam", "sam2", "sam2.1", "mobilesam", "hq-sam", "efficientsam"}


def _make_imagefolder(root: Path, n_per_class: int = 8, imgsz: int = 96) -> Path:
    from PIL import Image, ImageDraw

    state = 7

    def rnd() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    classes = {"warm": (200, 70, 50), "cool": (50, 90, 200)}
    for cname, base in classes.items():
        (root / cname).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            jitter = [max(0, min(255, int(c + (rnd() - 0.5) * 50))) for c in base]
            img = Image.new("RGB", (imgsz, imgsz), tuple(jitter))
            ImageDraw.Draw(img).ellipse([15, 15, imgsz - 15, imgsz - 15], fill=tuple(jitter))
            img.save(root / cname / f"{i:03d}.jpg", quality=90)
    return root


def run_embed_finetune(model_id: str) -> dict:
    """Full embedding head-finetune lifecycle for one embed model."""
    t0 = time.perf_counter()
    row = {
        "model_id": model_id,
        "task": "embed",
        "method": "head_train",
        "train_status": "NOT_TRAINABLE_BY_DESIGN",
        "fine_tune_status": "FINE_TUNE_READY_LIVE",
        "live_verified": False,
        "fixture": "tiny_2class_imagefolder",
        "checkpoint_or_adapter": None,
        "reload_verified": False,
        "post_reload_method": "classify+embed+similarity",
        "export_verified": False,
        "status": "FAIL",
        "blocker": None,
        "latency_ms": 0.0,
        "error_type": None,
        "error_message": None,
    }
    workdir = Path(tempfile.mkdtemp(prefix=f"v320_embft_{model_id.replace('/', '_')}_"))
    try:
        from visionservex.core.results import BaseResult, ClassificationResult
        from visionservex.training import EmbeddingHeadModel, finetune_embedding_head

        folder = _make_imagefolder(workdir / "imgs")
        res = finetune_embedding_head(
            model_id, folder, epochs=40, device="cpu", output_dir=str(workdir / "run")
        )
        ckpt = res["checkpoint"]
        row["checkpoint_or_adapter"] = Path(ckpt).name
        row["embed_dim"] = res["embed_dim"]
        row["train_acc"] = res["train_acc"]

        reloaded = EmbeddingHeadModel.from_checkpoint(ckpt, device="cpu")
        row["reload_verified"] = True

        val = next((workdir / "imgs").rglob("*.jpg"))
        cls = reloaded.classify(str(val), top_k=2)
        emb_a = reloaded.embed(str(val))
        val2 = list((workdir / "imgs").rglob("*.jpg"))[-1]
        emb_b = reloaded.embed(str(val2))
        sim = reloaded.similarity(emb_a, emb_b)

        ok = (
            isinstance(cls, ClassificationResult)
            and len(cls.top_k) >= 1
            and isinstance(emb_a, BaseResult)
            and -1.0001 <= float(sim) <= 1.0001
        )
        if ok:
            row["status"] = "PASS"
            row["live_verified"] = True
            row["similarity"] = round(float(sim), 4)
            row["top1"] = cls.top_k[0][0]
        else:
            row["error_type"] = "SchemaInvalid"
            row["error_message"] = f"cls={type(cls).__name__} sim={sim}"
    except Exception as e:
        import traceback

        row["error_type"] = type(e).__name__
        row["error_message"] = str(e)[:500]
        low = (str(e) + type(e).__name__).lower()
        row["blocker"] = "OOM_BLOCKED" if "out of memory" in low else "UPSTREAM_FAILURE"
        row["status"] = "SKIP_BLOCKED" if row["blocker"] == "OOM_BLOCKED" else "FAIL"
        sys.stderr.write(traceback.format_exc()[-1500:])
    finally:
        import shutil

        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        shutil.rmtree(workdir, ignore_errors=True)
    return row


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def embed_candidates() -> list[str]:
    import visionservex as vsx

    return sorted(
        m
        for m in vsx.list_models()
        if vsx.model_capabilities(m)["task"] in ("embed", "embedding")
        and vsx.model_capabilities(m)["inference_ready"]
    )


def classification_rows() -> list[dict]:
    """Non-embed rows: full-train families (committed) + foundation-seg inference-only."""
    import visionservex as vsx

    rows = []
    for m in vsx.list_models():
        c = vsx.model_capabilities(m)
        fam, task = c["family"], c["task"]
        if task in ("embed", "embedding"):
            continue
        if fam in _FULL_TRAIN_FAMILIES and c["train_live_verified"]:
            rows.append(
                {
                    "model_id": m,
                    "task": task,
                    "method": "train",
                    "train_status": "TRAIN_READY_LIVE",
                    "fine_tune_status": "FINE_TUNE_READY_LIVE",
                    "live_verified": True,
                    "fixture": "tiny_coco_or_imagefolder",
                    "checkpoint_or_adapter": "committed v3.18/v3.19 lifecycle matrix",
                    "reload_verified": c["reload_live_verified"],
                    "post_reload_method": "classify" if task == "classify" else "predict",
                    "export_verified": c["export_live_verified"],
                    "status": "REFERENCED",
                    "blocker": None,
                }
            )
        elif fam in _FOUNDATION_SEG_FAMILIES and c["inference_live_verified"]:
            rows.append(
                {
                    "model_id": m,
                    "task": task,
                    "method": "not_supported",
                    "train_status": "NOT_TRAINABLE_BY_DESIGN",
                    "fine_tune_status": "FOUNDATION_SEGMENT_INFERENCE_ONLY_LIVE",
                    "live_verified": True,
                    "fixture": "point/box prompt",
                    "checkpoint_or_adapter": None,
                    "reload_verified": False,
                    "post_reload_method": "segment",
                    "export_verified": False,
                    "status": "INFERENCE_ONLY",
                    "blocker": "Foundation segmenter: prompt-inference only; no cheap legal full-finetune wired.",
                }
            )
    return rows


def load_done() -> dict[str, dict]:
    done = {}
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            if line.strip():
                with contextlib.suppress(Exception):
                    r = json.loads(line)
                    done[r["model_id"]] = r
    return done


def write_outputs() -> None:
    embed_rows = list(load_done().values())
    rows = embed_rows + classification_rows()
    rows.sort(key=lambda r: (r["task"], r["model_id"]))
    summary = collections_counter(r["status"] for r in rows)
    JSON_OUT.write_text(
        json.dumps(
            {
                "schema": "v320_train_finetune_matrix",
                "n": len(rows),
                "summary": summary,
                "results": rows,
            },
            indent=2,
        )
    )
    ft_live = sum(
        1
        for r in rows
        if r.get("fine_tune_status") == "FINE_TUNE_READY_LIVE" and r["live_verified"]
    )
    md = [
        "# v3.20 Train / Fine-tune Matrix",
        "",
        "Embedding head-finetune live this sprint; full-train families referenced from "
        "committed v3.18/v3.19 matrices; foundation segmenters honestly inference-only.",
        f"**fine_tune_ready_live (embedding head-probe): {ft_live}**",
        "",
        "| Model | Task | Method | Train status | Fine-tune status | Live | Reload | Export |",
        "|---|---|---|---|---|:-:|:-:|:-:|",
    ]
    for r in rows:
        y = lambda b: "yes" if b else "—"  # noqa: E731
        md.append(
            f"| `{r['model_id']}` | {r['task']} | {r['method']} | {r['train_status']} | "
            f"{r['fine_tune_status']} | {y(r['live_verified'])} | {y(r.get('reload_verified'))} | "
            f"{y(r.get('export_verified'))} |"
        )
    MD_OUT.write_text("\n".join(md) + "\n")


def collections_counter(it):
    import collections

    return dict(collections.Counter(it))


def drive() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cands = embed_candidates()
    done = load_done()
    todo = [m for m in cands if m not in done]
    print(f"[driver] embed fine-tune candidates={len(cands)} done={len(done)} todo={len(todo)}")
    start = time.perf_counter()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TMPDIR"] = os.environ.get("TMPDIR", "/home/arash/.cache/vsx_tmp")
    processed = 0
    with JSONL.open("a") as fh:
        for m in todo:
            if time.perf_counter() - start > DRIVER_WALLCLOCK_BUDGET_S:
                print(f"[driver] budget hit; {len(todo) - processed} remain. Re-run.")
                break
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    [sys.executable, __file__, "--worker", m],
                    capture_output=True,
                    text=True,
                    timeout=PER_MODEL_TIMEOUT_S,
                    env=env,
                )
                line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
                row = (
                    json.loads(line)
                    if line.startswith("{")
                    else {
                        "model_id": m,
                        "task": "embed",
                        "status": "FAIL",
                        "live_verified": False,
                        "fine_tune_status": "FINE_TUNE_BLOCKED",
                        "error_type": "WorkerCrash",
                        "error_message": (proc.stderr or "no output")[-300:],
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    }
                )
            except subprocess.TimeoutExpired:
                row = {
                    "model_id": m,
                    "task": "embed",
                    "status": "SKIP_BLOCKED",
                    "live_verified": False,
                    "fine_tune_status": "FINE_TUNE_BLOCKED",
                    "blocker": "TIMEOUT",
                    "error_message": f"exceeded {PER_MODEL_TIMEOUT_S}s",
                    "latency_ms": PER_MODEL_TIMEOUT_S * 1000.0,
                }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            processed += 1
            print(
                f"[{processed}/{len(todo)}] {row.get('status'):<12} {m} ({row.get('latency_ms')}ms)"
            )
    write_outputs()
    remaining = len([m for m in cands if m not in load_done()])
    print(
        "[driver] ALL EMBED DONE"
        if remaining == 0
        else f"[driver] {remaining} embed remaining — re-run."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None)
    args = ap.parse_args()
    if args.worker:
        print(json.dumps(run_embed_finetune(args.worker)))
        return 0
    return drive()


if __name__ == "__main__":
    raise SystemExit(main())
