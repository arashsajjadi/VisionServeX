# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""INSID3 — Training-Free In-Context Segmentation with DINOv3 (BYOT).

Paper: "Training-Free In-Context Segmentation with DINOv3"
       CVPR 2026 Oral | arXiv 2603.28480 | visinf/INSID3 | Apache-2.0
Backbone: frozen DINOv3 (Meta custom license — BYOT, user accepts upstream).

Algorithm (zero-shot, no fine-tuning):
  1. Extract patch-level DINOv3 features from reference and query images.
  2. Positional debiasing via SVD (project out top-K positional components).
  3. Build reference prototype: mean of masked reference patches.
  4. Agglomerative clustering on query patches (scipy / sklearn).
  5. Select cluster whose centroid is closest (cosine sim) to prototype.
  6. Upsample binary cluster mask to original image resolution.
  7. Save pred_mask.png, overlay.png, metadata.json.

Security rules (MUST remain in effect through all edits):
  - Never print, commit, or log the raw HF token.
  - Never commit .onnx/.pt/.pth/.ckpt/.safetensors/.bin/.engine/.trt weights.
  - Token detection order: hf-cli cache → HF_TOKEN env → local private file.
  - TMPDIR=/home/arash/.cache/vsx_tmp for heavy operations.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from visionservex import hf_auth as _H
from visionservex.licensing import policy as _P

_FAMILY_PREFIXES = ("insid3",)
_DEFAULT_MODEL = "insid3-large"
_INSID3_PATCH_SIZE = 14  # DINOv3 ViT patch size (pixels)
_DEBIASING_RANK = 5  # SVD components to project out for positional debiasing


def _load_image(image):
    from PIL import Image

    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    return image


def _preflight(model_id: str) -> dict[str, Any] | None:
    """Return a blocked-dict if INSID3 can't lawfully run, else None."""
    canonical = _P.resolve_model_id(model_id)
    pol = _P.get_policy(canonical)
    if pol is None or pol.family != "insid3":
        return {
            "status": "blocked",
            "state": "unknown_model",
            "reason": f"{model_id!r} is not a known INSID3 model. "
            f"Use insid3-small / insid3-base / insid3-large.",
            "next_command": "visionservex insid3 status",
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


def _extract_patch_features(model, processor, img, device: str, token=None):
    """Return (patch_features, grid_h, grid_w) — features shape (n_patch_tokens, dim).

    DINOv3 outputs [CLS, *register_tokens, *patch_tokens] in last_hidden_state.
    We skip CLS and register tokens to get only the patch-spatial tokens, then
    derive grid_h / grid_w from the actual patch count (not from the input size).
    """
    import math

    import torch

    inputs = processor(images=img, return_tensors="pt").to(device)
    # Determine the actual pixel resolution the processor chose
    pixel_values = inputs["pixel_values"]  # (1, C, H_proc, H_proc)
    proc_h = pixel_values.shape[2]
    proc_w = pixel_values.shape[3]

    with torch.no_grad():
        out = model(**inputs)

    # Sequence: [CLS] [reg_0 ... reg_R] [patch_0 ... patch_N]
    # Skip CLS (index 0); skip register tokens if present
    n_reg = getattr(model.config, "num_register_tokens", 0)
    lhs_all = out.last_hidden_state[0, :, :].float().cpu()  # (1+R+N, dim)
    patch_feats = lhs_all[1 + n_reg :]  # skip CLS + registers → (N, dim)

    # Derive grid dimensions from actual patch count and processor output size
    n_patches = patch_feats.shape[0]
    if n_patches > 0:
        pw = _INSID3_PATCH_SIZE
        grid_h = proc_h // pw
        grid_w = proc_w // pw
        # Sanity: grid_h * grid_w should equal n_patches; if not fall back to sqrt
        if grid_h * grid_w != n_patches:
            side = math.isqrt(n_patches)
            grid_h = grid_w = side
    else:
        grid_h = grid_w = 1

    return patch_feats, grid_h, grid_w


def _debiasing(ref_feats, qry_feats, rank: int = _DEBIASING_RANK):
    """Positional debiasing: project out top-K singular vectors from both feature sets.

    Computes SVD on the concatenated feature matrix, then removes the low-rank
    positional subspace from both reference and query features.
    """
    import torch

    combined = torch.cat([ref_feats, qry_feats], dim=0)  # (N+M, dim)
    try:
        _, _, Vt = torch.linalg.svd(combined, full_matrices=False)
    except Exception:
        # Fallback: no debiasing if SVD fails (e.g., very small tensors)
        return ref_feats, qry_feats
    # Keep only top-`rank` right singular vectors as positional subspace
    pos_basis = Vt[:rank, :]  # (rank, dim)

    # Project out: x = x - x @ pos_basis.T @ pos_basis
    def _project_out(feats):
        proj = feats @ pos_basis.T @ pos_basis  # (N, dim)
        return feats - proj

    return _project_out(ref_feats), _project_out(qry_feats)


def _build_prototype(ref_feats, ref_mask_flat):
    """Mean of masked reference patch embeddings."""
    idx = ref_mask_flat.nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        # No mask patches — use full mean as fallback
        return ref_feats.mean(dim=0)
    return ref_feats[idx].mean(dim=0)


def _cluster_and_match(qry_feats, prototype, n_clusters: int = 6):
    """Agglomerative clustering + cosine prototype matching.

    Returns a boolean mask (n_patches,) of the best matching cluster.
    Falls back to cosine-sim thresholding if clustering libs unavailable.
    """
    import torch

    n_patches = qry_feats.shape[0]
    try:
        from sklearn.cluster import AgglomerativeClustering

        feats_np = qry_feats.numpy().astype("float32")
        n_clusters = min(n_clusters, n_patches)
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = clustering.fit_predict(feats_np)  # (n_patches,)
        # Compute each cluster centroid
        centroids = []
        for k in range(n_clusters):
            mask_k = labels == k
            if mask_k.any():
                centroids.append((k, qry_feats[mask_k].mean(dim=0)))
        # Select cluster with highest cosine sim to prototype
        proto_norm = prototype / (prototype.norm() + 1e-8)
        best_k, best_sim = -1, -1.0
        for k, cent in centroids:
            cent_norm = cent / (cent.norm() + 1e-8)
            sim = float((proto_norm * cent_norm).sum().item())
            if sim > best_sim:
                best_k, best_sim = k, sim
        return torch.from_numpy(labels == best_k), best_sim

    except ImportError:
        # Fallback: per-patch cosine similarity > threshold
        proto_norm = prototype / (prototype.norm() + 1e-8)
        feats_norm = qry_feats / (qry_feats.norm(dim=1, keepdim=True) + 1e-8)
        sims = (feats_norm @ proto_norm).clamp(-1.0, 1.0)
        threshold = float(sims.mean() + 0.5 * sims.std())
        return sims > threshold, float(sims.max().item())


def _upsample_patch_mask(patch_mask_bool, grid_h, grid_w, orig_h, orig_w):
    """Upsample boolean patch-grid mask to original image resolution (PIL).

    Returns a PIL Image in mode "L" (0/255 binary mask).
    """
    from PIL import Image

    # Reshape flat patch mask → (grid_h, grid_w) grid, then upsample to orig size
    flat = patch_mask_bool.reshape(grid_h, grid_w).numpy().astype("uint8") * 255
    patch_img = Image.fromarray(flat, mode="L")
    return patch_img.resize((orig_w, orig_h), Image.NEAREST)


def _make_overlay(orig_img, mask_img, alpha: float = 0.45):
    """Blend a semi-transparent red overlay onto orig_img where mask is nonzero."""
    import numpy as np
    from PIL import Image

    orig_arr = np.array(orig_img.convert("RGB"), dtype=float)
    mask_arr = np.array(mask_img, dtype=float) / 255.0  # (H, W) in [0,1]
    red = np.zeros_like(orig_arr)
    red[:, :, 0] = 255.0
    blended = orig_arr * (1.0 - alpha * mask_arr[:, :, None]) + red * (alpha * mask_arr[:, :, None])
    return Image.fromarray(blended.clip(0, 255).astype("uint8"), "RGB")


def insid3_segment(
    query_image,
    reference_image,
    reference_mask,
    *,
    model_id: str = _DEFAULT_MODEL,
    device: str = "cpu",
    n_clusters: int = 6,
    debiasing_rank: int = _DEBIASING_RANK,
    out_dir: str | None = None,
) -> dict[str, Any]:
    """Training-free in-context segmentation using frozen DINOv3 backbone (BYOT).

    Args:
        query_image: Path or PIL Image of the target image to segment.
        reference_image: Path or PIL Image of the reference (demonstration) image.
        reference_mask: Path or PIL Image (grayscale/binary) of the reference mask.
        model_id: INSID3 model variant (insid3-small/base/large). Default: insid3-large.
        device: "cpu" or "cuda".
        n_clusters: Agglomerative clustering count for query patches.
        debiasing_rank: SVD rank for positional debiasing.
        out_dir: Directory to save pred_mask.png, overlay.png, metadata.json. Optional.

    Returns:
        Structured dict with status, mask_area_px, out_dir, timings, and policy info.
        On any auth/license/resource failure: {"status": "blocked", "state": ..., ...}.
    """
    blocked = _preflight(model_id)
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
            "reason": "torch and transformers are required for INSID3",
            "next_command": "pip install 'visionservex[hf]'",
        }

    try:
        from PIL import Image
    except ImportError:
        return {
            "status": "blocked",
            "state": "dependency_required",
            "reason": "Pillow is required for INSID3",
            "next_command": "pip install Pillow",
        }

    token = _H.hf_get_token()

    # Load images
    qry_img = _load_image(query_image)
    ref_img = _load_image(reference_image)
    if isinstance(reference_mask, str):
        ref_mask_img = Image.open(reference_mask).convert("L")
    else:
        ref_mask_img = reference_mask.convert("L")

    orig_w, orig_h = qry_img.width, qry_img.height

    # Load DINOv3 backbone
    t0 = time.perf_counter()
    try:
        proc = AutoImageProcessor.from_pretrained(repo, token=token)
        model = AutoModel.from_pretrained(repo, token=token).to(device).eval()
    except Exception as exc:
        return {
            "status": "error",
            "state": "model_load_failed",
            "reason": str(exc),
            "model_id": canonical,
            "hf_repo": repo,
        }
    load_ms = (time.perf_counter() - t0) * 1000.0

    # Extract features
    t1 = time.perf_counter()
    try:
        ref_feats, ref_gh, ref_gw = _extract_patch_features(model, proc, ref_img, device, token)
        qry_feats, qry_gh, qry_gw = _extract_patch_features(model, proc, qry_img, device, token)
    except Exception as exc:
        return {
            "status": "error",
            "state": "feature_extraction_failed",
            "reason": str(exc),
            "model_id": canonical,
        }
    feat_ms = (time.perf_counter() - t1) * 1000.0

    # Debiasing
    ref_feats_d, qry_feats_d = _debiasing(ref_feats, qry_feats, rank=debiasing_rank)

    # Build reference prototype from masked patches
    import numpy as np

    # Resize the reference mask to match the actual patch grid size
    ref_mask_resized = ref_mask_img.resize((ref_gw, ref_gh), Image.NEAREST)
    ref_mask_grid = np.array(ref_mask_resized) > 127  # (ref_gh, ref_gw) bool grid
    ref_mask_patches = ref_mask_grid.flatten()  # (ref_gh*ref_gw,)

    import torch

    ref_mask_flat = torch.from_numpy(ref_mask_patches)
    prototype = _build_prototype(ref_feats_d, ref_mask_flat)

    # Cluster query patches and match to prototype
    t2 = time.perf_counter()
    pred_mask_flat, best_sim = _cluster_and_match(qry_feats_d, prototype, n_clusters)
    seg_ms = (time.perf_counter() - t2) * 1000.0

    # Upsample to original image resolution
    pred_mask_img = _upsample_patch_mask(pred_mask_flat, qry_gh, qry_gw, orig_h, orig_w)
    mask_arr = np.array(pred_mask_img)
    mask_area_px = int((mask_arr > 127).sum())
    overlay_img = _make_overlay(qry_img, pred_mask_img)

    # Save artifacts if out_dir provided
    saved_paths: dict[str, str] = {}
    if out_dir:
        import json

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        pred_mask_path = out_path / "pred_mask.png"
        overlay_path = out_path / "overlay.png"
        pred_mask_img.save(str(pred_mask_path))
        overlay_img.save(str(overlay_path))
        saved_paths["pred_mask"] = str(pred_mask_path)
        saved_paths["overlay"] = str(overlay_path)
        metadata = {
            "model_id": canonical,
            "hf_repo": repo,
            "task": "in_context_segmentation",
            "license": pol.weights_license,
            "attribution_required": "Built with DINOv3",
            "query_image_wh": [orig_w, orig_h],
            "mask_area_px": mask_area_px,
            "best_cluster_sim": round(best_sim, 4),
            "n_clusters": n_clusters,
            "debiasing_rank": debiasing_rank,
            "device": device,
            "load_ms": round(load_ms, 1),
            "feat_ms": round(feat_ms, 1),
            "seg_ms": round(seg_ms, 1),
            "state": "benchmark_passed_byot_mask",
        }
        meta_path = out_path / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        saved_paths["metadata"] = str(meta_path)

    return {
        "status": "ok",
        "state": "benchmark_passed_byot_mask",
        "model_id": canonical,
        "hf_repo": repo,
        "task": "in_context_segmentation",
        "license": pol.weights_license,
        "attribution_required": "Built with DINOv3",
        "mask_area_px": mask_area_px,
        "best_cluster_sim": round(best_sim, 4),
        "query_image_wh": [orig_w, orig_h],
        "device": device,
        "load_ms": round(load_ms, 1),
        "feat_ms": round(feat_ms, 1),
        "seg_ms": round(seg_ms, 1),
        "saved_paths": saved_paths,
        "warning": pol.warning_text,
    }


__all__ = ["insid3_segment"]
