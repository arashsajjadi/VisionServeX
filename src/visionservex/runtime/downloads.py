# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Model weight download utilities.

Supports four download backends, dispatched on ``ModelEntry.download_type``:

* ``huggingface``     — uses ``huggingface_hub.snapshot_download`` when the
                        ``hf`` extra is installed; falls back to an explicit
                        ``checkpoint_url`` if provided.
* ``github_release``  — same direct-URL path as ``direct_url`` with progress.
* ``direct_url``      — streams the URL with resume + SHA-256 verification.
* ``synthetic``       — built-in models; no-op success.
* ``manual``          — raises with upstream instructions; never auto-pulls.
* ``external_api``    — raises explaining the model is API-gated.
* ``not_available``   — raises with a clear message.

Tokens (e.g. ``HF_TOKEN``) are never written to logs.

We never download arbitrary URLs from user input; only URLs that already
appear in a registry entry are fetched.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from visionservex.config import get_settings
from visionservex.registry import ModelEntry
from visionservex.utils.hashing import sha256_file, verify_sha256
from visionservex.utils.logging import get_logger
from visionservex.utils.paths import ensure_dir

_log = get_logger(__name__)


class DownloadError(RuntimeError):
    """Raised for any download/verification failure with a human-friendly message."""


class ManualDownloadRequired(DownloadError):
    pass


class ExternalAPIModel(DownloadError):
    pass


class InsufficientDiskSpace(DownloadError):
    pass


@dataclass
class DownloadProgress:
    """Single progress event emitted by the downloader."""

    model_id: str
    phase: str  # "starting" | "downloading" | "verifying" | "loading" | "done" | "error"
    message: str = ""
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    started_at: float = 0.0

    @property
    def percent(self) -> float:
        if not self.total_bytes:
            return 0.0
        return min(100.0, (self.downloaded_bytes / self.total_bytes) * 100.0)

    @property
    def speed_bytes_per_sec(self) -> float:
        elapsed = max(1e-6, time.monotonic() - self.started_at)
        return self.downloaded_bytes / elapsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "phase": self.phase,
            "message": self.message,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "percent": round(self.percent, 1),
            "speed_bytes_per_sec": int(self.speed_bytes_per_sec),
        }


ProgressCallback = Callable[[DownloadProgress], None]

# Per-model download locks so two concurrent requests share one download.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _model_lock(model_id: str) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(model_id)
        if lock is None:
            lock = threading.Lock()
            _locks[model_id] = lock
        return lock


# =================================================================
# Paths and manifest
# =================================================================


def model_dir(entry: ModelEntry) -> Path:
    settings = get_settings()
    return ensure_dir(Path(settings.cache.cache_dir) / "models" / entry.id)


def manifest_path(entry: ModelEntry) -> Path:
    return model_dir(entry) / "manifest.json"


def write_manifest(entry: ModelEntry, payload: dict[str, Any]) -> Path:
    path = manifest_path(entry)
    base = {
        "id": entry.id,
        "license": entry.license,
        "license_uncertain": entry.license_uncertain,
        "upstream_url": entry.upstream_url,
        "download_type": entry.download_type,
        "downloaded_at": time.time(),
    }
    base.update(payload)
    path.write_text(json.dumps(base, indent=2, default=str), encoding="utf-8")
    return path


def read_manifest(entry: ModelEntry) -> dict | None:
    path = manifest_path(entry)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def cached_path(entry: ModelEntry) -> Path | None:
    """Return the path to a verified-cached primary checkpoint, or None."""
    if entry.download_type == "synthetic":
        return model_dir(entry)
    if entry.download_type == "package_managed":
        # For package-managed models, the manifest records the package cache.
        manifest = read_manifest(entry)
        if not manifest:
            return None
        saved_to = manifest.get("saved_to")
        if saved_to and Path(saved_to).exists():
            return Path(saved_to)
        return None
    if not entry.checkpoint_filename and not entry.checkpoint_url and not entry.hf_repo_id:
        return None
    manifest = read_manifest(entry)
    if not manifest:
        return None
    saved_to = manifest.get("saved_to")
    if not saved_to:
        return None
    p = Path(saved_to)
    if not p.exists():
        return None
    if entry.checkpoint_sha256 and not verify_sha256(p, entry.checkpoint_sha256):
        return None
    return p


def is_cached(entry: ModelEntry) -> bool:
    return cached_path(entry) is not None


# =================================================================
# Public API
# =================================================================


def download(
    entry: ModelEntry,
    *,
    progress: ProgressCallback | None = None,
    force: bool = False,
    offline: bool | None = None,
    require_auto_download: bool = False,
    max_size_gb: float | None = None,
) -> Path:
    """Download ``entry`` to the local cache. Returns the primary file path.

    Thread- and process-safe: concurrent calls share one underlying download.
    """
    settings = get_settings()
    offline = settings.cache.offline if offline is None else offline

    if require_auto_download and not entry.auto_download:
        raise DownloadError(
            f"Model {entry.id!r} is not allowed for auto-download by the registry. "
            f"Run `visionservex pull {entry.id}` explicitly, or pick a model with "
            f"auto_download=true. See `visionservex recommend`."
        )

    if entry.download_type == "synthetic":
        return _emit_done(
            entry,
            progress,
            message="built-in mock model; no download required",
            path=model_dir(entry),
        )

    if entry.download_type == "package_managed":
        return _download_package_managed(entry, progress)

    if entry.download_type == "manual":
        raise ManualDownloadRequired(
            f"Model {entry.id!r} cannot be downloaded automatically.\n"
            f"Reason: upstream requires manual install or custom code.\n"
            f"Instructions: {entry.upstream_url}\n"
            f"After installing/extracting, place the weight file under: {model_dir(entry)}\n"
            f"Then run `visionservex cache repair {entry.id}` to register it."
        )

    if entry.download_type == "external_api":
        raise ExternalAPIModel(
            f"Model {entry.id!r} is provided as an external API and cannot be self-hosted "
            f"by VisionServeX. Review upstream terms at: {entry.upstream_url}"
        )

    if entry.download_type == "not_available":
        raise DownloadError(
            f"Model {entry.id!r} has no registered download method. "
            f"Run `visionservex info {entry.id}` for details."
        )

    if offline:
        cached = cached_path(entry)
        if cached:
            return cached
        raise DownloadError(
            f"offline mode is enabled and no cached checkpoint exists for {entry.id!r}"
        )

    lock = _model_lock(entry.id)
    with lock:
        if not force:
            cached = cached_path(entry)
            if cached:
                return _emit_done(entry, progress, message="cached", path=cached)

        # Disk space sanity check
        if entry.size_bytes:
            free = shutil.disk_usage(str(model_dir(entry).parent)).free
            if free < entry.size_bytes * 1.5:
                raise InsufficientDiskSpace(
                    f"insufficient disk space: model {entry.id!r} needs ~{entry.size_bytes / 1e9:.2f} GB; "
                    f"only {free / 1e9:.2f} GB free at {model_dir(entry).parent}"
                )
        if (
            max_size_gb is not None
            and entry.size_bytes is not None
            and entry.size_bytes / (1024**3) > max_size_gb
        ):
            raise DownloadError(
                f"model {entry.id!r} exceeds max_size_gb limit "
                f"({entry.size_bytes / (1024**3):.2f} > {max_size_gb})"
            )

        # Dispatch by type
        if entry.download_type == "huggingface":
            return _download_huggingface(entry, progress)
        if entry.download_type in {"direct_url", "github_release"}:
            return _download_direct(entry, progress)
        if entry.download_type == "package_managed":
            return _download_package_managed(entry, progress)

        raise DownloadError(f"unknown download_type {entry.download_type!r} for {entry.id!r}")


def _emit(progress: ProgressCallback | None, event: DownloadProgress) -> None:
    if progress is not None:
        try:
            progress(event)
        except Exception:  # pragma: no cover
            _log.debug("progress callback raised; ignoring", exc_info=True)


def _emit_done(
    entry: ModelEntry, progress: ProgressCallback | None, *, message: str, path: Path
) -> Path:
    _emit(
        progress,
        DownloadProgress(
            model_id=entry.id,
            phase="done",
            message=message,
            started_at=time.monotonic(),
        ),
    )
    return path


# =================================================================
# Direct URL / GitHub release
# =================================================================


def _download_direct(entry: ModelEntry, progress: ProgressCallback | None) -> Path:
    if not entry.checkpoint_url:
        raise DownloadError(
            f"model {entry.id!r} declares download_type={entry.download_type!r} but no checkpoint_url"
        )
    settings = get_settings()
    target_dir = model_dir(entry)
    filename = entry.checkpoint_filename or Path(entry.checkpoint_url).name
    final = target_dir / filename
    tmp = final.with_suffix(final.suffix + ".partial")
    existing_bytes = tmp.stat().st_size if tmp.exists() else 0
    chunk = settings.cache.download_chunk_bytes
    started = time.monotonic()
    _emit(
        progress,
        DownloadProgress(
            model_id=entry.id,
            phase="starting",
            message=f"GET {entry.checkpoint_url}",
            started_at=started,
            total_bytes=entry.size_bytes,
        ),
    )

    headers: dict[str, str] = {}
    if existing_bytes:
        headers["Range"] = f"bytes={existing_bytes}-"
    token = os.environ.get("HF_TOKEN") if "huggingface.co" in entry.checkpoint_url else None
    if token:
        headers["Authorization"] = f"Bearer {token}"

    attempt = 0
    backoff = 1.0
    last_exc: Exception | None = None
    while attempt < 4:
        attempt += 1
        try:
            with httpx.stream(
                "GET",
                entry.checkpoint_url,
                timeout=60.0,
                headers=headers,
                follow_redirects=True,
            ) as resp:
                if resp.status_code not in (200, 206):
                    raise DownloadError(
                        f"download failed: HTTP {resp.status_code} for {entry.checkpoint_url}"
                    )
                total = entry.size_bytes
                cl = resp.headers.get("Content-Length")
                if cl:
                    with contextlib.suppress(ValueError):
                        total = int(cl) + (existing_bytes if resp.status_code == 206 else 0)
                mode = "ab" if existing_bytes and resp.status_code == 206 else "wb"
                downloaded = existing_bytes if mode == "ab" else 0
                with tmp.open(mode) as fh:
                    for piece in resp.iter_bytes(chunk_size=chunk):
                        if not piece:
                            continue
                        fh.write(piece)
                        downloaded += len(piece)
                        _emit(
                            progress,
                            DownloadProgress(
                                model_id=entry.id,
                                phase="downloading",
                                downloaded_bytes=downloaded,
                                total_bytes=total,
                                started_at=started,
                            ),
                        )
            break
        except (httpx.HTTPError, DownloadError) as exc:
            last_exc = exc
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
            existing_bytes = tmp.stat().st_size if tmp.exists() else 0
            headers["Range"] = (
                f"bytes={existing_bytes}-" if existing_bytes else headers.pop("Range", None)
            )
    else:
        raise DownloadError(f"download failed after retries: {last_exc}")

    if entry.checkpoint_sha256:
        _emit(
            progress,
            DownloadProgress(
                model_id=entry.id,
                phase="verifying",
                message="checking SHA-256",
                started_at=started,
            ),
        )
        if not verify_sha256(tmp, entry.checkpoint_sha256):
            tmp.unlink(missing_ok=True)
            raise DownloadError(f"SHA-256 mismatch for {entry.id!r}; partial file removed")

    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), str(final))
    write_manifest(
        entry,
        {
            "saved_to": str(final),
            "size_bytes": final.stat().st_size,
            "sha256": entry.checkpoint_sha256 or sha256_file(final),
            "source_url": entry.checkpoint_url,
        },
    )
    return _emit_done(entry, progress, message="saved", path=final)


# =================================================================
# Hugging Face
# =================================================================


def _download_huggingface(entry: ModelEntry, progress: ProgressCallback | None) -> Path:
    """Use ``huggingface_hub.snapshot_download`` when available."""
    if not entry.hf_repo_id:
        raise DownloadError(
            f"model {entry.id!r} declares download_type=huggingface but hf_repo_id is missing"
        )
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        raise DownloadError(
            f"huggingface_hub is required to download {entry.id!r}. "
            f"Install with: pip install 'visionservex[hf]'"
        ) from exc

    get_settings()
    target_dir = model_dir(entry)
    started = time.monotonic()
    _emit(
        progress,
        DownloadProgress(
            model_id=entry.id,
            phase="starting",
            message=f"snapshot_download {entry.hf_repo_id}",
            started_at=started,
            total_bytes=entry.size_bytes,
        ),
    )

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    kwargs: dict[str, Any] = {
        "repo_id": entry.hf_repo_id,
        "local_dir": str(target_dir / "snapshot"),
    }
    if entry.hf_revision:
        kwargs["revision"] = entry.hf_revision
    if token:
        kwargs["token"] = token

    try:
        local = Path(snapshot_download(**kwargs))
    except Exception as exc:
        # Token-free log message; never echo the secret.
        raise DownloadError(
            f"Hugging Face download failed for {entry.id!r}: {exc}.\n"
            f"Check that the repo {entry.hf_repo_id!r} exists and is public, "
            f"or set HF_TOKEN if it is gated."
        ) from exc

    # Primary file: prefer explicit filename; otherwise the largest weights file.
    primary: Path | None = None
    if entry.checkpoint_filename:
        candidate = local / entry.checkpoint_filename
        if candidate.exists():
            primary = candidate
    if primary is None:
        weight_globs = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.onnx"]
        candidates: list[Path] = []
        for pattern in weight_globs:
            candidates.extend(local.rglob(pattern))
        if candidates:
            primary = max(candidates, key=lambda p: p.stat().st_size)
    if primary is None:
        primary = local  # directory snapshot — engine resolves further.

    size = sum(p.stat().st_size for p in local.rglob("*") if p.is_file())
    write_manifest(
        entry,
        {
            "saved_to": str(primary),
            "snapshot_dir": str(local),
            "size_bytes": size,
            "hf_repo_id": entry.hf_repo_id,
            "hf_revision": entry.hf_revision,
            "files": sorted(str(p.relative_to(local)) for p in local.rglob("*") if p.is_file()),
        },
    )
    return _emit_done(entry, progress, message="downloaded from Hugging Face", path=primary)


# =================================================================
# Package-managed download (e.g. rfdetr)
# =================================================================


def _download_package_managed(entry: ModelEntry, progress: ProgressCallback | None) -> Path:
    """Trigger a package-managed model download by instantiating its engine.

    Models like RF-DETR manage their own checkpoint download inside the package
    (``maybe_download_pretrain_weights``). We call a lightweight probe function
    that triggers the download, then write a manifest pointing at the package's
    own cache so ``is_cached`` returns True on subsequent calls.
    """
    started = time.monotonic()
    _emit(
        progress,
        DownloadProgress(
            model_id=entry.id,
            phase="starting",
            message=f"package-managed download for {entry.id}",
            started_at=started,
        ),
    )

    # The engine module provides a ``_trigger_package_download`` helper.
    try:
        from visionservex.engines.rfdetr import trigger_package_download

        cache_path = trigger_package_download(entry)
    except ImportError as exc:
        raise DownloadError(
            f"engine module for {entry.id!r} is not available. "
            f"Install with: pip install 'visionservex[{entry.install_extra or 'rfdetr'}]'"
        ) from exc

    write_manifest(
        entry,
        {
            "saved_to": str(cache_path),
            "download_type": "package_managed",
            "package": entry.install_extra or "unknown",
        },
    )
    _emit(
        progress,
        DownloadProgress(
            model_id=entry.id,
            phase="done",
            message=f"weights in {cache_path}",
            started_at=started,
        ),
    )
    return cache_path


# =================================================================
# Cache management
# =================================================================


def cache_root() -> Path:
    return ensure_dir(Path(get_settings().cache.cache_dir) / "models")


def cache_listing() -> list[dict[str, Any]]:
    root = cache_root()
    out: list[dict[str, Any]] = []
    if not root.exists():
        return out
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        size = sum(p.stat().st_size for p in sub.rglob("*") if p.is_file())
        manifest = None
        m = sub / "manifest.json"
        if m.exists():
            try:
                manifest = json.loads(m.read_text(encoding="utf-8"))
            except Exception:
                manifest = None
        out.append(
            {
                "model_id": sub.name,
                "path": str(sub),
                "size_bytes": size,
                "manifest": manifest,
            }
        )
    return out


def cache_clean(model_id: str | None = None) -> int:
    root = cache_root()
    if not root.exists():
        return 0
    freed = 0
    targets = [root / model_id] if model_id else list(root.iterdir())
    for target in targets:
        if target.exists() and target.is_dir():
            freed += sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
            shutil.rmtree(target)
    return freed


def cache_verify(model_id: str | None = None) -> list[dict[str, Any]]:
    """Verify checksums and manifest of cached models. Returns per-model report."""
    from visionservex.registry import default_registry

    reg = default_registry()
    targets = []
    if model_id:
        try:
            targets.append(reg.get(model_id))
        except Exception:
            return [{"model_id": model_id, "ok": False, "reason": "not in registry"}]
    else:
        targets = list(reg.list())

    report: list[dict[str, Any]] = []
    for entry in targets:
        cached = cached_path(entry)
        if cached is None and entry.download_type != "synthetic":
            continue
        ok = True
        reason = "ok"
        if (
            entry.checkpoint_sha256
            and cached
            and cached.is_file()
            and not verify_sha256(cached, entry.checkpoint_sha256)
        ):
            ok = False
            reason = "sha256 mismatch"
        report.append(
            {
                "model_id": entry.id,
                "ok": ok,
                "reason": reason,
                "path": str(cached) if cached else None,
            }
        )
    return report


def cache_repair(model_id: str) -> bool:
    """Re-scan a model's cache directory and rebuild its manifest if possible."""
    from visionservex.registry import RegistryError, default_registry

    try:
        entry = default_registry().get(model_id)
    except RegistryError:
        return False

    d = model_dir(entry)
    if not d.exists():
        return False

    # Best-effort: find largest weight file under the directory
    candidates: list[Path] = []
    for ext in ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.onnx", "*.pkl"):
        candidates.extend(d.rglob(ext))
    if not candidates:
        return False
    primary = max(candidates, key=lambda p: p.stat().st_size)
    write_manifest(
        entry,
        {
            "saved_to": str(primary),
            "size_bytes": sum(p.stat().st_size for p in d.rglob("*") if p.is_file()),
            "repaired": True,
        },
    )
    return True


__all__ = [
    "DownloadError",
    "DownloadProgress",
    "ExternalAPIModel",
    "InsufficientDiskSpace",
    "ManualDownloadRequired",
    "ProgressCallback",
    "cache_clean",
    "cache_listing",
    "cache_repair",
    "cache_root",
    "cache_verify",
    "cached_path",
    "download",
    "is_cached",
    "manifest_path",
    "model_dir",
    "read_manifest",
    "write_manifest",
]
