"""Path utilities with traversal protection."""

from __future__ import annotations

import os
from pathlib import Path


class PathTraversalError(ValueError):
    """Raised when a user-provided path resolves outside the allowed root."""


def safe_join(root: Path | str, candidate: str) -> Path:
    """Join ``candidate`` onto ``root`` and reject traversal.

    The result must resolve to a path inside ``root``. Symlinks pointing
    outside the root are rejected.
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        root_path.mkdir(parents=True, exist_ok=True)

    if candidate is None or candidate == "":
        raise PathTraversalError("empty path")

    raw = Path(candidate)
    if raw.is_absolute() or any(part == ".." for part in raw.parts):
        raise PathTraversalError(f"unsafe path: {candidate!r}")

    final = (root_path / raw).resolve()
    try:
        final.relative_to(root_path)
    except ValueError as exc:
        raise PathTraversalError(f"path {candidate!r} escapes root {root_path}") from exc
    return final


def ensure_dir(path: Path | str) -> Path:
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_within(child: Path | str, parent: Path | str) -> bool:
    """Return True if ``child`` resolves under ``parent`` (no resolution required)."""
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def writeable(path: Path | str) -> bool:
    try:
        return os.access(Path(path), os.W_OK)
    except OSError:
        return False


__all__ = ["PathTraversalError", "safe_join", "ensure_dir", "is_within", "writeable"]
