# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tests for the model downloader.

Uses an in-process HTTP server so CI does not download real model weights.
"""

from __future__ import annotations

import http.server
import socket
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from visionservex.registry import ModelEntry, default_registry
from visionservex.runtime.downloads import (
    DownloadError,
    ManualDownloadRequired,
    cache_clean,
    cache_verify,
    download,
    is_cached,
    model_dir,
)
from visionservex.utils.hashing import sha256_file

# ---------- test server fixture ----------


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


@contextmanager
def _http_server(payload: bytes):
    """Run a tiny HTTP server that streams ``payload`` for any GET."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            range_header = self.headers.get("Range")
            start = 0
            if range_header and range_header.startswith("bytes="):
                start = int(range_header.split("=", 1)[1].split("-")[0] or 0)
            body = payload[start:]
            status = 206 if start > 0 else 200
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/octet-stream")
            if start > 0:
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
                )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args, **kwargs):  # silence
            pass

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = _Server(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/weights.bin"
    finally:
        server.shutdown()


def _entry(url: str, *, sha: str | None = None, size: int | None = None) -> ModelEntry:
    return ModelEntry(
        id="test-direct",
        display_name="Test Direct",
        task="detect",
        family="test",
        engine="mock",
        backend="mock",
        license="Apache-2.0",
        upstream_url="https://example.com/test",
        download_type="direct_url",
        checkpoint_url=url,
        checkpoint_filename="weights.bin",
        checkpoint_sha256=sha,
        size_bytes=size,
        auto_download=True,
    )


# ---------- tests ----------


def test_synthetic_models_need_no_download(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    from visionservex.config import reload_settings

    reload_settings()
    reg = default_registry()
    mock_entry = reg.get("mock-detect")
    path = download(mock_entry)
    assert path.exists()


def test_direct_download_succeeds(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    from visionservex.config import reload_settings

    reload_settings()

    payload = b"hello-vsx-weights" * 100
    import hashlib

    sha = hashlib.sha256(payload).hexdigest()

    with _http_server(payload) as url:
        entry = _entry(url, sha=sha, size=len(payload))
        path = download(entry)
        assert path.exists()
        assert path.read_bytes() == payload
        assert sha256_file(path) == sha
        assert is_cached(entry)


def test_direct_download_sha_mismatch_removes_file(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    from visionservex.config import reload_settings

    reload_settings()

    payload = b"abcdef" * 200
    bad_sha = "0" * 64
    with _http_server(payload) as url:
        entry = _entry(url, sha=bad_sha, size=len(payload))
        with pytest.raises(DownloadError, match="SHA-256 mismatch"):
            download(entry)
        # tmp file removed
        partial = model_dir(entry) / "weights.bin.partial"
        assert not partial.exists()


def test_resume_download(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    from visionservex.config import reload_settings

    reload_settings()
    payload = b"PYTHON-IS-FUN!" * 500
    with _http_server(payload) as url:
        entry = _entry(url, size=len(payload))
        # Seed a partial file
        d = model_dir(entry)
        d.mkdir(parents=True, exist_ok=True)
        partial = d / "weights.bin.partial"
        partial.write_bytes(payload[: len(payload) // 3])
        path = download(entry)
        assert path.read_bytes() == payload


def test_offline_without_cache_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VISIONSERVEX_CACHE__OFFLINE", "true")
    from visionservex.config import reload_settings

    reload_settings()
    entry = _entry("http://127.0.0.1:1/never", size=100)
    with pytest.raises(DownloadError, match="offline"):
        download(entry)


def test_manual_download_raises_friendly():
    reg = default_registry()
    entry = reg.get("co-dino-inst-vit-l-coco")
    with pytest.raises(ManualDownloadRequired):
        download(entry)


def test_external_api_model_raises_friendly():
    reg = default_registry()
    entry = reg.get("grounding-dino-1.5")
    with pytest.raises(DownloadError):
        download(entry)


def test_cache_verify_returns_report(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    from visionservex.config import reload_settings

    reload_settings()
    payload = b"verify-me" * 50
    import hashlib

    sha = hashlib.sha256(payload).hexdigest()
    with _http_server(payload) as url:
        entry = _entry(url, sha=sha, size=len(payload))
        # Manually register so cache_verify can find it
        from visionservex.registry import default_registry

        reg = default_registry()
        reg.register(entry, replace=True)
        download(entry)
        report = cache_verify(entry.id)
        assert report and report[0]["ok"]


def test_cache_clean_frees_bytes(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    from visionservex.config import reload_settings

    reload_settings()
    payload = b"clean-me" * 1000
    with _http_server(payload) as url:
        entry = _entry(url, size=len(payload))
        download(entry)
        freed = cache_clean(entry.id)
        assert freed > 0


def test_require_auto_download_blocks_non_auto(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    from visionservex.config import reload_settings

    reload_settings()
    payload = b"abc" * 10
    with _http_server(payload) as url:
        entry = _entry(url, size=len(payload))
        entry.auto_download = False
        with pytest.raises(DownloadError, match="not allowed for auto-download"):
            download(entry, require_auto_download=True)


def test_duplicate_download_shares_lock(monkeypatch, tmp_path):
    """Two concurrent downloads should not both write the same file."""
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    from visionservex.config import reload_settings

    reload_settings()
    payload = b"shared-lock-payload" * 200
    import hashlib

    sha = hashlib.sha256(payload).hexdigest()
    results = []
    with _http_server(payload) as url:
        entry = _entry(url, sha=sha, size=len(payload))

        def _go():
            results.append(download(entry))

        threads = [threading.Thread(target=_go) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    paths = {str(p) for p in results}
    assert len(paths) == 1  # same file path
    assert Path(results[0]).read_bytes() == payload
