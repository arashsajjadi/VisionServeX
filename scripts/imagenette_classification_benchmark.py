#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""v2.51.0 Imagenette classification benchmark.

Evaluates classification models on Imagenette val split, mapping
ImageNet-1K top-k predictions to the 10 Imagenette WNIDs.

Imagenette WNIDs and their canonical ImageNet-1K class names (for label matching):
  n01440764  tench
  n02102040  English springer
  n02979186  cassette player
  n03000684  chain saw / chainsaw
  n03028079  church
  n03394916  French horn
  n03417042  garbage truck
  n03425413  gas pump
  n03445777  golf ball
  n03888257  parachute
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

IMAGENETTE_VAL = "/home/arash/datasets/imagenette2-320/val"
IMAGES_PER_CLASS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
OUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v251_cls.json"
MODELS_ARG = sys.argv[3].split(",") if len(sys.argv) > 3 else None

DEFAULT_MODELS = [
    "convnextv2-tiny",
    "convnextv2-base",
    "convnextv2-large",
    "maxvit-tiny-tf-224",
    "swinv2-tiny",
    "swinv2-small",
    "swinv2-base",
    "swinv2-large",
]

# Standard ImageNet-1K class index for each Imagenette WNID.
# Used when model returns LABEL_N format (no id2label mapping configured).
IMAGENET1K_INDEX: dict[str, int] = {
    "n01440764": 0,  # tench
    "n02102040": 217,  # English springer spaniel
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}
IMAGENET1K_INDEX_TO_WNID = {v: k for k, v in IMAGENET1K_INDEX.items()}

# Imagenette WNID → list of keywords to match in ImageNet-1K label (case-insensitive)
WNID_KEYWORDS: dict[str, list[str]] = {
    "n01440764": ["tench"],
    "n02102040": ["springer", "english springer"],
    "n02979186": ["cassette player"],
    "n03000684": ["chain saw", "chainsaw"],
    "n03028079": ["church"],
    "n03394916": ["french horn"],
    "n03417042": ["garbage truck", "dustcart"],
    "n03425413": ["gas pump", "gasoline pump", "petrol pump"],
    "n03445777": ["golf ball"],
    "n03888257": ["parachute", "chute"],
}

WNID_NAMES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def _label_matches_wnid(label: str, wnid: str) -> bool:
    """Check if an ImageNet-1K label corresponds to an Imagenette WNID.

    Handles two label formats:
    1. Text labels like 'tench, Tinca tinca' — keyword match against WNID_KEYWORDS.
    2. 'LABEL_N' format (when model lacks id2label) — index lookup via IMAGENET1K_INDEX.
    """
    label_lower = label.lower()
    # Handle LABEL_N format
    if label_lower.startswith("label_"):
        try:
            idx = int(label_lower.replace("label_", ""))
            return IMAGENET1K_INDEX_TO_WNID.get(idx) == wnid
        except ValueError:
            return False
    return any(kw in label_lower for kw in WNID_KEYWORDS.get(wnid, []))


def _prepare_val_set(max_per_class: int) -> list[tuple[str, str]]:
    """Return list of (image_path, wnid) pairs for Imagenette val split."""
    pairs = []
    for wnid in sorted(WNID_KEYWORDS.keys()):
        wnid_dir = Path(IMAGENETTE_VAL) / wnid
        if not wnid_dir.exists():
            print(f"  WARNING: {wnid_dir} not found")
            continue
        imgs = sorted(wnid_dir.glob("*.JPEG"))[:max_per_class]
        pairs.extend((str(img), wnid) for img in imgs)
    return pairs


def _run_model(model_id: str, val_set: list[tuple[str, str]]) -> dict:
    from PIL import Image

    from visionservex import VisionModel

    result = {
        "model_id": model_id,
        "status": "failed",
        "code": "",
        "dataset": "imagenette2-320 val",
        "dataset_source": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
        "wnid_mapping": WNID_NAMES,
        "top1": None,
        "top5": None,
        "macro_accuracy": None,
        "per_class_accuracy": None,
        "latency_ms_p50": None,
        "fps": None,
        "n_evaluated": len(val_set),
        "error": None,
    }

    try:
        model = VisionModel(model_id, auto_pull=False)
        print(f"  [{model_id}] loaded")
    except Exception as e:
        result["code"] = "MODEL_LOAD_FAILED"
        result["error"] = str(e)[:300]
        print(f"  [{model_id}] LOAD FAILED: {str(e)[:80]}")
        return result

    correct_top1, correct_top5 = 0, 0
    per_class_correct: dict[str, int] = defaultdict(int)
    per_class_total: dict[str, int] = defaultdict(int)
    latencies: list[float] = []

    for img_path, true_wnid in val_set:
        try:
            img = Image.open(img_path).convert("RGB")
            t0 = time.perf_counter()
            out = model.predict(img)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            top_k = out.top_k  # list of (label_str, score)
            # Handle NaN scores: sort by score safely, treating NaN as 0
            import math

            top_k = sorted(top_k, key=lambda x: x[1] if not math.isnan(x[1]) else 0.0, reverse=True)
            per_class_total[true_wnid] += 1
            # Top-1
            if top_k and _label_matches_wnid(top_k[0][0], true_wnid):
                correct_top1 += 1
                per_class_correct[true_wnid] += 1
            # Top-5
            if any(_label_matches_wnid(lbl, true_wnid) for lbl, _ in top_k[:5]):
                correct_top5 += 1
        except Exception:
            continue

    try:
        del model
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    n = len(val_set)
    if n == 0:
        result["code"] = "NO_IMAGES"
        return result

    per_cls_acc = {
        wnid: round(per_class_correct.get(wnid, 0) / max(per_class_total.get(wnid, 1), 1), 4)
        for wnid in WNID_KEYWORDS
    }
    macro_acc = sum(per_cls_acc.values()) / len(per_cls_acc)
    lats = sorted(latencies)
    p50 = lats[len(lats) // 2] if lats else None
    fps = 1000.0 / (sum(latencies) / len(latencies)) if latencies else None

    result.update(
        {
            "status": "ok",
            "code": "OK",
            "top1": round(correct_top1 / n, 4),
            "top5": round(correct_top5 / n, 4),
            "macro_accuracy": round(macro_acc, 4),
            "per_class_accuracy": {WNID_NAMES[k]: v for k, v in per_cls_acc.items()},
            "latency_ms_p50": round(p50, 1) if p50 else None,
            "fps": round(fps, 1) if fps else None,
        }
    )
    print(
        f"  [{model_id}] top1={result['top1']:.4f} top5={result['top5']:.4f} "
        f"macro={result['macro_accuracy']:.4f} lat={p50:.0f}ms"
    )
    return result


def main():
    val_set = _prepare_val_set(IMAGES_PER_CLASS)
    models_to_run = MODELS_ARG or DEFAULT_MODELS
    print(f"Imagenette classification benchmark: {len(models_to_run)} models")
    print(f"Val set: {len(val_set)} images ({IMAGES_PER_CLASS} per class × 10 classes)")
    print(f"Dataset: {IMAGENETTE_VAL}")

    results = []
    for i, mid in enumerate(models_to_run):
        print(f"\n[{i + 1}/{len(models_to_run)}] {mid}")
        r = _run_model(mid, val_set)
        results.append(r)

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "benchmark_type": "imagenette_classification",
                "dataset": "imagenette2-320 val",
                "wnid_mapping": WNID_NAMES,
                "images_per_class": IMAGES_PER_CLASS,
                "n_images": len(val_set),
                "models": results,
            },
            f,
            indent=2,
        )

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\nSummary: {len(ok)}/{len(results)} succeeded")
    if ok:
        print(f"Best top1: {max(r['top1'] for r in ok):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
