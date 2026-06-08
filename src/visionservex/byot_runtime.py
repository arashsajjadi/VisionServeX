# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""BYOT runtime for gated models (SAM 3 / SAM 3.1 + DINOv3).

These models are gated and license-required. This module runs them **only** with
the user's own token, after the upstream license is accepted, loading weights from
the user's Hugging Face cache. Nothing here ships, mirrors, or logs weights or
tokens.

Backends:
* DINOv3  -> ``transformers.AutoModel`` / ``AutoImageProcessor`` (image embedding).
* SAM 3   -> ``transformers.Sam3Model`` / ``Sam3Processor`` (concept/text prompt
  instance segmentation).

Every public function returns a structured dict. On any license/auth/resource
blocker it returns ``{"status": "blocked", "state": ..., "reason": ..., "next_command": ...}``
— it never fabricates a result.
"""

from __future__ import annotations

import time
from typing import Any

from visionservex import hf_auth as _H
from visionservex.licensing import policy as _P


def _load_image(image):
    from PIL import Image

    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    return image


def _preflight(model_id: str, family_prefixes: tuple[str, ...]) -> dict[str, Any] | None:
    """Return a blocked-dict if the model can't lawfully run, else None."""
    canonical = _P.resolve_model_id(model_id)
    pol = _P.get_policy(canonical)
    if pol is None or not any(canonical.startswith(p) for p in family_prefixes):
        return {
            "status": "blocked",
            "state": "unknown_model",
            "reason": f"{model_id} is not a BYOT model handled here",
            "next_command": f"visionservex model license {model_id}",
        }
    try:
        _H.hf_require_user_accepted_license(canonical)
    except _H.HFLicenseError as exc:
        return {
            "status": "blocked",
            "state": exc.state,
            "model_id": canonical,
            "reason": str(exc),
            "next_command": exc.next_command,
            "warning": pol.warning_text,
        }
    return None


def dinov3_embed(model_id: str, image, *, device: str = "cpu") -> dict[str, Any]:
    """Compute a DINOv3 image embedding (BYOT). Returns shape/norm metadata only."""
    blocked = _preflight(model_id, ("dinov3",))
    if blocked:
        return blocked
    canonical = _P.resolve_model_id(model_id)
    pol = _P.get_policy(canonical)
    repo = pol.hf_repo
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        return {
            "status": "blocked",
            "state": "dependency_required",
            "reason": "transformers/torch not installed",
            "next_command": "pip install 'visionservex[hf]'",
        }
    token = _H.hf_get_token()
    img = _load_image(image)
    t0 = time.perf_counter()
    proc = AutoImageProcessor.from_pretrained(repo, token=token)
    model = AutoModel.from_pretrained(repo, token=token).to(device).eval()
    load_ms = (time.perf_counter() - t0) * 1000.0
    inputs = proc(images=img, return_tensors="pt").to(device)
    t1 = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs)
    infer_ms = (time.perf_counter() - t1) * 1000.0
    # pooled embedding: prefer pooler_output, else mean over last_hidden_state
    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        pooled = out.last_hidden_state.mean(dim=1)
    pooled = pooled.float().cpu()
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "status": "ok",
        "state": "benchmark_passed_byot",
        "model_id": canonical,
        "hf_repo": repo,
        "task": "embedding",
        "license": pol.weights_license,
        "device": device,
        "embedding_dim": int(pooled.shape[-1]),
        "embedding_norm": float(pooled.norm().item()),
        "last_hidden_state_shape": list(out.last_hidden_state.shape),
        "params_millions": round(n_params / 1e6, 2),
        "load_ms": round(load_ms, 1),
        "infer_ms": round(infer_ms, 1),
        "warning": pol.warning_text,
    }


def sam3_segment(
    model_id: str, image, *, text: str | None = None, device: str = "cpu"
) -> dict[str, Any]:
    """Run SAM 3 / SAM 3.1 concept/text-prompt instance segmentation (BYOT)."""
    blocked = _preflight(model_id, ("sam3",))
    if blocked:
        return blocked
    canonical = _P.resolve_model_id(model_id)
    pol = _P.get_policy(canonical)
    repo = pol.hf_repo
    if not text:
        return {
            "status": "blocked",
            "state": "prompt_required",
            "reason": "SAM 3 requires a text/concept prompt (e.g. text='person')",
            "next_command": f"VSX.sam('{canonical}').segment(image, text='person')",
        }
    try:
        import torch
        from transformers import Sam3Model, Sam3Processor
    except ImportError:
        return {
            "status": "blocked",
            "state": "dependency_required",
            "reason": "transformers Sam3 classes not available",
            "next_command": "pip install -U 'visionservex[hf]'  # needs transformers>=5.3",
        }
    token = _H.hf_get_token()
    img = _load_image(image)
    t0 = time.perf_counter()
    proc = Sam3Processor.from_pretrained(repo, token=token)
    # SAM3.1 uses a non-standard weight filename (sam3.1_multiplex.pt).
    # Try direct HF id first; on OSError fall back to snapshot + local alias.
    try:
        model = Sam3Model.from_pretrained(repo, token=token).to(device).eval()
    except OSError:
        import shutil
        from huggingface_hub import snapshot_download

        local_snap = snapshot_download(repo, token=token)
        pt_files = list(__import__("pathlib").Path(local_snap).glob("*.pt"))
        if not pt_files:
            raise
        working = __import__("pathlib").Path(local_snap).parent / "_vsx_sam3_working"
        working.mkdir(exist_ok=True)
        for cfg in ("config.json", "processor_config.json", "tokenizer.json",
                    "tokenizer_config.json", "special_tokens_map.json",
                    "merges.txt", "vocab.json"):
            src = __import__("pathlib").Path(local_snap) / cfg
            if src.exists():
                shutil.copy(src, working / cfg)
        alias = working / "pytorch_model.bin"
        if not alias.exists():
            alias.symlink_to(pt_files[0])
        model = Sam3Model.from_pretrained(str(working)).to(device).eval()
    load_ms = (time.perf_counter() - t0) * 1000.0
    inputs = proc(images=img, text=text, return_tensors="pt").to(device)
    t1 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    infer_ms = (time.perf_counter() - t1) * 1000.0
    # post-process to instance masks (API name varies across transformers builds)
    n_masks = None
    scores: list[float] = []
    try:
        target = [(img.height, img.width)]
        results = None
        for fn in ("post_process_instance_segmentation", "post_process_grounded_object_detection"):
            if hasattr(proc, fn):
                try:
                    results = getattr(proc, fn)(outputs, target_sizes=target, threshold=0.4)
                    break
                except TypeError:
                    results = getattr(proc, fn)(outputs)
                    break
        if results:
            r0 = results[0]
            for key in ("masks", "segmentation", "boxes"):
                val = r0.get(key) if isinstance(r0, dict) else None
                if val is not None:
                    n_masks = len(val)
                    break
            sc = r0.get("scores") if isinstance(r0, dict) else None
            if sc is not None:
                scores = [round(float(s), 4) for s in list(sc)[:10]]
    except Exception as exc:
        return {
            "status": "ok",
            "state": "benchmark_passed_byot",
            "model_id": canonical,
            "hf_repo": repo,
            "task": "concept_segmentation",
            "text": text,
            "license": pol.weights_license,
            "device": device,
            "load_ms": round(load_ms, 1),
            "infer_ms": round(infer_ms, 1),
            "postprocess_note": f"forward pass OK; post-process API mismatch: {type(exc).__name__}",
            "warning": pol.warning_text,
        }
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "status": "ok",
        "state": "benchmark_passed_byot",
        "model_id": canonical,
        "hf_repo": repo,
        "task": "concept_segmentation",
        "text": text,
        "license": pol.weights_license,
        "device": device,
        "num_masks": n_masks,
        "scores": scores,
        "params_millions": round(n_params / 1e6, 2),
        "load_ms": round(load_ms, 1),
        "infer_ms": round(infer_ms, 1),
        "warning": pol.warning_text,
    }


__all__ = ["dinov3_embed", "sam3_segment"]
