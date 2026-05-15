"""File hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO, Callable


_CHUNK = 1024 * 1024


def sha256_file(path: Path | str, progress: Callable[[int], None] | None = None) -> str:
    """Stream a file through SHA-256 and return the hex digest."""
    hasher = hashlib.sha256()
    p = Path(path)
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            hasher.update(chunk)
            if progress:
                progress(len(chunk))
    return hasher.hexdigest()


def sha256_stream(fh: BinaryIO) -> str:
    hasher = hashlib.sha256()
    for chunk in iter(lambda: fh.read(_CHUNK), b""):
        hasher.update(chunk)
    return hasher.hexdigest()


def verify_sha256(path: Path | str, expected: str) -> bool:
    """Return True if the file's SHA-256 matches ``expected`` (case-insensitive)."""
    return sha256_file(path).lower() == expected.lower().strip()


__all__ = ["sha256_file", "sha256_stream", "verify_sha256"]
