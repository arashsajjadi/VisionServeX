# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Result classes for embedding / retrieval / search tasks.

These extend BaseResult so they fit into the existing predict() pipeline
but expose vector-specific fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from visionservex.core.results import BaseResult


@dataclass(kw_only=True)
class EmbeddingResult(BaseResult):
    """A single image embedding."""

    kind: str = "embedding"  # type: ignore[assignment]
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    embedding_dim: int = 0
    normalized: bool = True

    def to_dict(self) -> dict[str, Any]:
        # Exclude the raw numpy array from JSON; expose stats only.
        base = {
            "kind": self.kind,
            "model_id": self.model_id,
            "task": self.task,
            "device": self.device,
            "precision": self.precision,
            "backend": self.backend,
            "latency_ms": self.latency_ms,
            "image_size": list(self.image_size),
            "embedding_dim": self.embedding_dim,
            "normalized": self.normalized,
            "embedding_norm": float(np.linalg.norm(self.embedding))
            if self.embedding.size > 0
            else 0.0,
            "embedding_first_5": self.embedding[:5].tolist() if self.embedding.size >= 5 else [],
            "warnings": self.warnings,
        }
        return base

    def summary(self) -> str:
        return (
            f"<EmbeddingResult model={self.model_id} dim={self.embedding_dim} "
            f"latency={self.latency_ms:.1f}ms>"
        )

    def save_npy(self, path: str) -> None:
        """Save the embedding as a numpy .npy file."""
        np.save(path, self.embedding)


@dataclass
class SimilarityResult:
    """Pairwise similarity between two embeddings."""

    model_id: str
    image_a: str
    image_b: str
    cosine_similarity: float
    embedding_dim: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "image_a": self.image_a,
            "image_b": self.image_b,
            "cosine_similarity": self.cosine_similarity,
            "embedding_dim": self.embedding_dim,
        }


@dataclass
class SearchHit:
    """One nearest-neighbor result."""

    image_path: str
    score: float
    rank: int


@dataclass
class SearchResult:
    """Top-k nearest neighbors for a query."""

    model_id: str
    query: str
    top_k: int
    hits: list[SearchHit] = field(default_factory=list)
    index_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "query": self.query,
            "top_k": self.top_k,
            "index_path": self.index_path,
            "hits": [
                {"rank": h.rank, "image_path": h.image_path, "score": h.score} for h in self.hits
            ],
        }


@dataclass
class DatasetReport:
    """Summary statistics about a folder of images, computed via embeddings."""

    folder: str
    model_id: str
    n_images: int = 0
    mean_pairwise_similarity: float = 0.0
    n_likely_duplicates: int = 0
    duplicate_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    suggested_clusters: int = 0
    diversity_score: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "folder": self.folder,
            "model_id": self.model_id,
            "n_images": self.n_images,
            "mean_pairwise_similarity": self.mean_pairwise_similarity,
            "n_likely_duplicates": self.n_likely_duplicates,
            "duplicate_pairs": [
                {"a": a, "b": b, "score": s} for a, b, s in self.duplicate_pairs[:20]
            ],
            "suggested_clusters": self.suggested_clusters,
            "diversity_score": self.diversity_score,
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Dataset Report — {self.folder}",
            "",
            f"**Model:** `{self.model_id}`  ",
            f"**Images:** {self.n_images}  ",
            f"**Mean pairwise similarity:** {self.mean_pairwise_similarity:.3f}  ",
            f"**Likely duplicates (sim > 0.98):** {self.n_likely_duplicates}  ",
            f"**Diversity score:** {self.diversity_score:.3f} (lower = more redundant)  ",
            f"**Suggested clusters (kmeans on PCA-32):** {self.suggested_clusters}  ",
            "",
        ]
        if self.duplicate_pairs:
            lines.append("## Top likely duplicate pairs")
            for a, b, s in self.duplicate_pairs[:10]:
                lines.append(f"- `{a}` ↔ `{b}` (sim={s:.4f})")
            lines.append("")
        if self.notes:
            lines.append("## Notes")
            for note in self.notes:
                lines.append(f"- {note}")
        return "\n".join(lines)


__all__ = [
    "DatasetReport",
    "EmbeddingResult",
    "SearchHit",
    "SearchResult",
    "SimilarityResult",
]
