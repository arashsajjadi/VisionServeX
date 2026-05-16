# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Embedding indexing / search / deduplication / dataset intelligence.

Uses numpy nearest-neighbor by default. Optional: sklearn for k-means.
No hard FAISS / hnswlib dependency.

Memory rules:
- Embeddings are saved to disk in .npy + .json manifest.
- Embeddings are NOT kept on GPU.
- Process isolation supported via VisionModel.predict(..., unload_after=True).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


@dataclass
class EmbeddingIndex:
    """A flat embedding index saved to disk.

    Layout:
        index_dir/
          embeddings.npy   (N, D) float32
          manifest.json    {model_id, image_paths, normalized}
    """

    index_dir: Path
    embeddings: np.ndarray  # shape (N, D)
    image_paths: list[str]
    model_id: str
    normalized: bool = True

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / "embeddings.npy", self.embeddings)
        manifest = {
            "model_id": self.model_id,
            "n_images": len(self.image_paths),
            "embedding_dim": int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0,
            "normalized": self.normalized,
            "image_paths": self.image_paths,
        }
        (self.index_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path) -> EmbeddingIndex:
        manifest_path = index_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest.json in {index_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        embeddings = np.load(index_dir / "embeddings.npy")
        return cls(
            index_dir=index_dir,
            embeddings=embeddings,
            image_paths=manifest["image_paths"],
            model_id=manifest["model_id"],
            normalized=manifest.get("normalized", True),
        )


def discover_images(folder: Path, max_images: int | None = None) -> list[Path]:
    """Return image paths under ``folder`` (recursive)."""
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in _IMG_EXTS)
    if max_images is not None:
        paths = paths[:max_images]
    return paths


def embed_folder(
    model_id: str,
    folder: Path,
    *,
    device: str = "auto",
    max_images: int | None = None,
    unload_after: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Compute embeddings for every image in ``folder``.

    Returns (embeddings_NxD, image_paths). Embeddings are L2-normalized.
    """
    from PIL import Image as _PIL

    from visionservex.core.embedding_results import EmbeddingResult
    from visionservex.core.model import VisionModel

    img_paths = discover_images(folder, max_images=max_images)
    if not img_paths:
        return np.zeros((0, 0), dtype=np.float32), []

    embeddings: list[np.ndarray] = []
    saved_paths: list[str] = []

    model = VisionModel(model_id, device=device)
    try:
        model._ensure_loaded()
        for ip in img_paths:
            try:
                img = _PIL.open(ip).convert("RGB")
                result = model.predict(img)
                if isinstance(result, EmbeddingResult) and result.embedding.size > 0:
                    embeddings.append(result.embedding.astype(np.float32))
                    saved_paths.append(str(ip))
            except Exception:
                continue
    finally:
        if unload_after:
            model.unload()

    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32), []

    return np.stack(embeddings, axis=0), saved_paths


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def search_index(
    index: EmbeddingIndex,
    query_embedding: np.ndarray,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Find top-k nearest neighbors in ``index`` for ``query_embedding``.

    Assumes both are L2-normalized → cosine similarity = dot product.
    """
    if index.embeddings.size == 0 or query_embedding.size == 0:
        return []
    sims = index.embeddings @ query_embedding  # shape (N,)
    k = min(top_k, len(sims))
    top_indices = np.argsort(-sims)[:k]
    return [(index.image_paths[i], float(sims[i])) for i in top_indices]


def deduplicate_index(
    index: EmbeddingIndex,
    threshold: float = 0.98,
) -> list[tuple[str, str, float]]:
    """Return pairs of likely duplicates (sim >= threshold)."""
    pairs: list[tuple[str, str, float]] = []
    if index.embeddings.shape[0] < 2:
        return pairs
    sim = index.embeddings @ index.embeddings.T  # (N, N)
    n = sim.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                pairs.append((index.image_paths[i], index.image_paths[j], float(sim[i, j])))
    pairs.sort(key=lambda x: -x[2])
    return pairs


def build_dataset_report(
    model_id: str,
    folder: Path,
    *,
    duplicate_threshold: float = 0.98,
    max_images: int | None = None,
    device: str = "auto",
) -> Any:  # DatasetReport, but avoid circular import in signature
    """Build a dataset report from a folder."""
    from visionservex.core.embedding_results import DatasetReport

    embeddings, paths = embed_folder(model_id, folder, device=device, max_images=max_images)
    if embeddings.shape[0] == 0:
        return DatasetReport(
            folder=str(folder), model_id=model_id, n_images=0, notes=["No images found."]
        )

    n = embeddings.shape[0]
    sim = embeddings @ embeddings.T
    # Mean off-diagonal similarity
    mask = ~np.eye(n, dtype=bool)
    mean_sim = float(sim[mask].mean()) if mask.any() else 0.0

    # Duplicate pairs
    dup_pairs: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= duplicate_threshold:
                dup_pairs.append((paths[i], paths[j], float(sim[i, j])))
    dup_pairs.sort(key=lambda x: -x[2])

    # Diversity = 1 - mean_sim
    diversity = max(0.0, 1.0 - mean_sim)

    # Suggested clusters: heuristic by sqrt(n)
    suggested_clusters = max(2, int(np.sqrt(n)))

    notes = []
    if mean_sim > 0.85:
        notes.append("High mean similarity — dataset may be redundant. Consider deduplication.")
    if len(dup_pairs) > n * 0.1:
        notes.append(
            f"More than 10% of pairs exceed similarity {duplicate_threshold}. Run deduplicate."
        )

    return DatasetReport(
        folder=str(folder),
        model_id=model_id,
        n_images=n,
        mean_pairwise_similarity=round(mean_sim, 4),
        n_likely_duplicates=len(dup_pairs),
        duplicate_pairs=dup_pairs,
        suggested_clusters=suggested_clusters,
        diversity_score=round(diversity, 4),
        notes=notes,
    )


def active_learning_select(
    model_id: str,
    folder: Path,
    *,
    budget: int = 100,
    device: str = "auto",
    max_images: int | None = None,
) -> list[str]:
    """Select diverse samples from a folder via farthest-point sampling on embeddings."""
    embeddings, paths = embed_folder(model_id, folder, device=device, max_images=max_images)
    if embeddings.shape[0] == 0:
        return []
    n = embeddings.shape[0]
    budget = min(budget, n)

    # Farthest-point sampling
    selected_idx: list[int] = [0]
    min_dists = np.zeros(n, dtype=np.float32)
    # 1 - cosine = distance for normalized embeddings
    dist_to_first = 1.0 - (embeddings @ embeddings[0])
    min_dists = dist_to_first.copy()
    min_dists[0] = -np.inf  # exclude

    for _ in range(budget - 1):
        next_idx = int(np.argmax(min_dists))
        if min_dists[next_idx] <= 0:
            break
        selected_idx.append(next_idx)
        dist_to_new = 1.0 - (embeddings @ embeddings[next_idx])
        min_dists = np.minimum(min_dists, dist_to_new)
        min_dists[next_idx] = -np.inf

    return [paths[i] for i in selected_idx]


def domain_shift_report(
    model_id: str,
    train_folder: Path,
    test_folder: Path,
    *,
    device: str = "auto",
    max_images: int | None = 200,
) -> dict[str, Any]:
    """Estimate train→test domain shift via mean embedding distance."""
    train_emb, _train_paths = embed_folder(
        model_id, train_folder, device=device, max_images=max_images
    )
    test_emb, _test_paths = embed_folder(
        model_id, test_folder, device=device, max_images=max_images
    )

    if train_emb.shape[0] == 0 or test_emb.shape[0] == 0:
        return {
            "status": "skip",
            "reason": "one or both folders have no images",
            "n_train": train_emb.shape[0],
            "n_test": test_emb.shape[0],
        }

    train_centroid = train_emb.mean(axis=0)
    test_centroid = test_emb.mean(axis=0)
    train_centroid /= max(np.linalg.norm(train_centroid), 1e-8)
    test_centroid /= max(np.linalg.norm(test_centroid), 1e-8)
    centroid_sim = float(np.dot(train_centroid, test_centroid))
    centroid_distance = 1.0 - centroid_sim

    # Distance: mean of test→nearest-train distances
    sim_xt = test_emb @ train_emb.T  # (N_test, N_train)
    max_per_test = sim_xt.max(axis=1)
    mean_nearest = float(max_per_test.mean())

    shift_score = 1.0 - mean_nearest
    if shift_score < 0.05:
        verdict = "low_shift"
    elif shift_score < 0.15:
        verdict = "moderate_shift"
    else:
        verdict = "high_shift"

    return {
        "status": "ok",
        "model_id": model_id,
        "n_train": int(train_emb.shape[0]),
        "n_test": int(test_emb.shape[0]),
        "centroid_similarity": round(centroid_sim, 4),
        "centroid_distance": round(centroid_distance, 4),
        "test_to_train_mean_nearest_similarity": round(mean_nearest, 4),
        "domain_shift_score": round(shift_score, 4),
        "verdict": verdict,
        "note": (
            "shift_score = 1 - mean(test→nearest-train cosine similarity). "
            "Lower is better (train/test distribution closer)."
        ),
    }


__all__ = [
    "EmbeddingIndex",
    "active_learning_select",
    "build_dataset_report",
    "cosine_similarity",
    "deduplicate_index",
    "discover_images",
    "domain_shift_report",
    "embed_folder",
    "search_index",
]
