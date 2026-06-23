#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Standalone MedSAM2 real-runtime smoke (END STATE A/B evidence).

This script is INTENTIONALLY standalone: it imports ONLY the upstream MedSAM2/SAM2
stack (`sam2`, `torch`, `numpy`, `PIL`) — never VisionServeX — so it can run inside
an isolated conda env (python 3.12 + torch 2.5.1) where the upstream is installed.

It proves, with machine-readable JSON, exactly how far the real MedSAM2 runtime gets:
import -> config -> checkpoint -> model build -> device -> eval -> 2D inference -> mask.

Every failure is a *structured* status, never a bare stack trace, so the output is a
reproducible record of either END STATE A (loaded/ran) or END STATE B (exact blocker).

MedSAM2 weights are RESEARCH/EDUCATION ONLY (non-commercial) — see HF wanglab/MedSAM2.
This script never asserts commercial use and always reports commercial_safe=false.

Usage (inside the isolated env):
    python real_runtime_smoke.py --checkpoint /path/MedSAM2_latest.pt \
        --config configs/sam2.1/sam2.1_hiera_t.yaml --device cpu --out /tmp/medsam2_out
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def _emit(payload: dict) -> None:
    print("MEDSAM2_SMOKE_JSON " + json.dumps(payload, default=str))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--image", default="", help="optional PNG/JPEG; else a synthetic 256x256")
    ap.add_argument("--box", default="64,64,192,192", help="x1,y1,x2,y2 box prompt")
    ap.add_argument("--out", default="/tmp/medsam2_smoke")
    args = ap.parse_args()

    result: dict = {
        "model_id": "medsam2",
        "runtime_type": "in_process_sam2_fork",
        "device": args.device,
        "config_path": args.config,
        "checkpoint_path": args.checkpoint,
        "commercial_safe": False,
        "license_note": "MedSAM2 weights are research/education only (non-commercial).",
        "status": "unknown",
        "warnings": [],
    }

    # --- dependency import ---
    try:
        import numpy as np
        import torch
        from PIL import Image

        result["torch_version"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        result["status"] = "MEDSAM2_REQUIRED"
        result["error"] = f"core deps missing: {exc}"
        _emit(result)
        return 3

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        result["status"] = "MEDSAM2_REQUIRED"
        result["error"] = f"sam2 not importable: {exc}"
        result["next_step"] = "pip install -e . from github.com/bowang-lab/MedSAM2 (provides sam2)"
        _emit(result)
        return 3

    # --- checkpoint presence ---
    ckpt = Path(args.checkpoint)
    result["checkpoint_detected"] = ckpt.is_file()
    if not ckpt.is_file():
        result["status"] = "MEDSAM2_CHECKPOINT_REQUIRED"
        result["error"] = f"checkpoint not found: {ckpt}"
        result["next_step"] = (
            "huggingface_hub.hf_hub_download('wanglab/MedSAM2','MedSAM2_latest.pt')"
        )
        _emit(result)
        return 4

    # --- build + load ---
    # MedSAM2 trains at 512 res; try its config first, then vanilla sam2.1 tiny.
    candidate_configs = [
        args.config,
        "configs/sam2.1_hiera_t512.yaml",
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "configs/sam2.1/sam2.1_hiera_t512.yaml",
    ]
    seen: list[str] = []
    t0 = time.perf_counter()
    model = None
    last_cfg_err = ""
    for cfg in candidate_configs:
        if cfg in seen:
            continue
        seen.append(cfg)
        try:
            model = build_sam2(cfg, str(ckpt), device=args.device)
            model.eval()
            result["config_path"] = cfg
            break
        except Exception as exc:
            last_cfg_err = f"{cfg}: {exc}"
            # a state_dict/runtime error means the config resolved but load failed
            if not isinstance(exc, FileNotFoundError) and "config" not in str(exc).lower():
                result["status"] = "MEDSAM2_CHECKPOINT_INVALID"
                result["error"] = f"checkpoint failed to load into model: {exc}"
                result["traceback"] = traceback.format_exc()[-1500:]
                _emit(result)
                return 6
            continue
    if model is None:
        result["status"] = "MEDSAM2_CONFIG_REQUIRED"
        result["error"] = f"no candidate config resolved. last: {last_cfg_err}"
        result["tried_configs"] = seen
        _emit(result)
        return 5
    result["checkpoint_validated"] = True
    result["load_time_seconds"] = round(time.perf_counter() - t0, 3)
    result["status"] = "loaded"

    # --- 2D inference ---
    try:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.image and Path(args.image).is_file():
            img = np.array(Image.open(args.image).convert("RGB"))
        else:
            # synthetic 256x256 with a bright square in the prompted region
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            img[64:192, 64:192] = 200
            result["warnings"].append("used synthetic 256x256 image (no --image given)")
        box = np.array([float(x) for x in args.box.split(",")], dtype=float)

        predictor = SAM2ImagePredictor(model)
        ti = time.perf_counter()
        with torch.inference_mode():
            predictor.set_image(img)
            masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=False)
        result["infer_time_seconds"] = round(time.perf_counter() - ti, 3)

        mask = np.asarray(masks)
        # SAM2 returns (N,H,W) or (H,W); normalize to 2D for the first mask
        m2d = mask[0] if mask.ndim == 3 else mask
        m2d = (m2d > 0).astype(np.uint8)
        Image.fromarray(m2d * 255).save(out_dir / "medsam2_mask_000.png")

        result["status"] = "inference_ok"
        result["input_mode"] = "2d_slice"
        result["prompt_type"] = "box"
        result["mask_shape"] = list(m2d.shape)
        result["mask_pixels_on"] = int(m2d.sum())
        result["n_masks"] = int(mask.shape[0]) if mask.ndim == 3 else 1
        result["scores"] = [float(s) for s in np.asarray(scores).ravel().tolist()]
        result["mask_path"] = str(out_dir / "medsam2_mask_000.png")
        if result["mask_pixels_on"] == 0:
            result["warnings"].append("empty mask — inference ran but produced no foreground")
            result["status"] = "inference_empty_mask"
        if args.device == "cpu" and torch.cuda.is_available():
            try:
                import resource

                result["peak_memory_mb"] = round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
                )
            except Exception:
                pass
    except Exception as exc:
        result["status"] = "MEDSAM2_RUNTIME_UNAVAILABLE"
        result["error"] = f"inference failed after load: {exc}"
        result["traceback"] = traceback.format_exc()[-1500:]
        _emit(result)
        return 7

    _emit(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
