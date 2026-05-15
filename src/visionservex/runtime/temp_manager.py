# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Secure temporary file management.

All temp files created by VisionServeX for uploaded images or intermediate
processing are managed here.  They are:
- Created with restricted permissions (0600).
- Deleted immediately after use or on error.
- Never written if privacy.save_inputs=False (default).
"""

from __future__ import annotations

import contextlib
import os
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path

from visionservex.utils.logging import get_logger

_log = get_logger(__name__)


@contextlib.contextmanager
def secure_temp_file(
    suffix: str = ".tmp",
    prefix: str = "vsx_",
    tmpdir: str | None = None,
    cleanup: bool = True,
) -> Iterator[Path]:
    """Create a temp file with 0600 permissions; delete on exit.

    Usage::

        with secure_temp_file(suffix=".jpg") as path:
            path.write_bytes(image_bytes)
            # ... use path ...
        # file is deleted here

    The file is always deleted on context exit, even if an exception occurs.
    """
    fd, fpath = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=tmpdir)
    try:
        # Restrict to owner read/write only
        os.chmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        os.close(fd)
        yield Path(fpath)
    finally:
        if cleanup:
            try:
                os.unlink(fpath)
            except FileNotFoundError:
                pass
            except Exception as exc:
                _log.warning("Failed to clean temp file %s: %s", fpath, exc)


def cleanup_temp_dir(tmpdir: str | None = None) -> int:
    """Remove all vsx_* temp files in the system temp dir.  Returns count removed."""
    base = Path(tmpdir or tempfile.gettempdir())
    removed = 0
    for p in base.glob("vsx_*"):
        try:
            if p.is_file():
                p.unlink()
                removed += 1
        except Exception as exc:
            _log.debug("Could not remove %s: %s", p, exc)
    return removed


def inspect_temp_dir(tmpdir: str | None = None) -> list[dict]:
    """List vsx_* temp files without revealing their contents."""
    base = Path(tmpdir or tempfile.gettempdir())
    result = []
    for p in base.glob("vsx_*"):
        try:
            s = p.stat()
            result.append(
                {
                    "path": str(p),
                    "size_bytes": s.st_size,
                    "created": s.st_ctime,
                }
            )
        except Exception:
            pass
    return result


__all__ = ["cleanup_temp_dir", "inspect_temp_dir", "secure_temp_file"]
