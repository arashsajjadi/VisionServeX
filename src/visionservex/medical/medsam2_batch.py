# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""MedSAM2 batch runner — order-preserving, deterministic outputs, honest manifest.

Loads ONE MedSAM2 model and runs it over a list of 2D inputs. The upstream
SAM2 image predictor holds per-image state on a single shared model, so a shared
model is NOT thread-safe: when a single model is shared we force sequential
execution (``effective_workers=1``) and record that in the manifest. GPU always
stays at one worker (never duplicate a giant model on one device). True parallel
behaviour of the executor itself is validated separately with a mocked predictor.

Deterministic output naming (never overwrites without ``overwrite=True``):
    {index:05d}_{stem}_{model_id}_mask_{mask_index:03d}.png
    {index:05d}_{stem}_{model_id}.json
plus a batch manifest ``medsam2_batch_manifest.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from visionservex.medical.medsam2_runtime import (
    MedSAM2RuntimeError,
    load_2d_input,
    load_medsam2_runtime,
    segment_2d,
)
from visionservex.medical.parallel import run_ordered

_MODEL_ID = "medsam2"


def _save_item(result: Any, index: int, stem: str, out_dir: Path, overwrite: bool) -> dict:
    import numpy as np
    from PIL import Image

    json_path = out_dir / f"{index:05d}_{stem}_{_MODEL_ID}.json"
    if json_path.exists() and not overwrite:
        raise MedSAM2RuntimeError(
            "OUTPUT_EXISTS", f"output exists: {json_path} (pass overwrite=True)"
        )
    masks = []
    for m, seg in enumerate(result.segments):
        mp = out_dir / f"{index:05d}_{stem}_{_MODEL_ID}_mask_{m:03d}.png"
        if mp.exists() and not overwrite:
            raise MedSAM2RuntimeError("OUTPUT_EXISTS", f"output exists: {mp} (pass overwrite=True)")
        Image.fromarray((np.asarray(seg.mask) * 255).astype(np.uint8)).save(mp)
        masks.append({"mask_path": str(mp), "score": seg.score})
    item_payload = {
        "index": index,
        "model_id": _MODEL_ID,
        "n_masks": len(masks),
        "masks": masks,
        "commercial_safe": False,
        "research_only": True,
    }
    json_path.write_text(json.dumps(item_payload, indent=2, default=str))
    return {"json_path": str(json_path), "n_masks": len(masks), "extra": {"masks": masks}}


def run_medsam2_batch(
    inputs: list[str],
    *,
    checkpoint: str | Path,
    config: str | None = None,
    device: str = "cpu",
    out_dir: str | Path,
    workers: int = 1,
    continue_on_error: bool = True,
    overwrite: bool = False,
) -> dict:
    """Run MedSAM2 over ``inputs`` (order-preserving). Returns a manifest dict.

    Raises :class:`MedSAM2RuntimeError` only for whole-batch setup failures (model
    load); per-item failures are captured in the manifest, never raised.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Whole-batch setup: load once. A load failure is a hard, structured error.
    rt = load_medsam2_runtime(checkpoint, config=config or None, device=device)

    # Shared single model is not thread-safe; GPU must not duplicate models.
    requested_workers = workers
    effective_workers = 1
    warnings: list[str] = []
    if requested_workers > 1:
        warnings.append(
            f"requested workers={requested_workers} clamped to 1: a single shared "
            "MedSAM2 model is not thread-safe (use process isolation for true parallelism)."
        )

    def _segment_fn(path: str, index: int) -> dict:
        img = load_2d_input(path)
        result = segment_2d(rt, img, boxes=None, slice_index=None)
        stem = Path(path).stem
        return _save_item(result, index, stem, out, overwrite)

    t0 = time.perf_counter()
    item_results = run_ordered(
        inputs,
        _segment_fn,
        workers=effective_workers,
        continue_on_error=continue_on_error,
    )
    elapsed = round(time.perf_counter() - t0, 3)

    per_item = []
    for r in item_results:
        row = r.to_dict()
        if r.status == "ok" and isinstance(r.value, dict):
            row["json_path"] = r.value.get("json_path")
            row["n_masks"] = r.value.get("n_masks")
        per_item.append(row)

    manifest = {
        "model_id": _MODEL_ID,
        "engine": "medsam2_runtime",
        "checkpoint": str(checkpoint),
        "config": rt.config_path,
        "device": device,
        "requested_workers": requested_workers,
        "effective_workers": effective_workers,
        "n_inputs": len(inputs),
        "n_ok": sum(1 for r in item_results if r.status == "ok"),
        "n_failed": sum(1 for r in item_results if r.status == "failed"),
        "n_skipped": sum(1 for r in item_results if r.status == "skipped"),
        "elapsed_seconds": elapsed,
        "items": per_item,
        "commercial_safe": False,
        "research_only": True,
        "warnings": warnings,
        "disclaimer": "Research/education only — NOT for diagnosis.",
    }
    (out / "medsam2_batch_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    manifest["manifest_path"] = str(out / "medsam2_batch_manifest.json")
    return manifest


__all__ = ["run_medsam2_batch"]
