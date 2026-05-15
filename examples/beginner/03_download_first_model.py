# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Beginner example 03 — download the recommended beginner model.

By default this picks the top recommendation for the `detect` task using
`recommend(simple=True)`. The mock model is fine if no real backend is
installed.
"""

from __future__ import annotations

from visionservex.runtime.downloads import (
    DownloadError,
    DownloadProgress,
    ManualDownloadRequired,
    download,
)
from visionservex.runtime.recommendations import recommend


def main() -> None:
    recs = recommend(task="detect", simple=True, limit=3)
    if not recs:
        print("No recommendations available.")
        return

    entry = recs[0].entry
    print(f"Recommended: {entry.id} (status={entry.status}, impl={entry.implementation_status})")
    print(f"License: {entry.license}")
    print(f"Download type: {entry.download_type}")
    if not entry.auto_download and entry.download_type != "synthetic":
        print(f"\nThis model is not auto-downloadable. See: {entry.upstream_url}")
        return

    def _cb(ev: DownloadProgress) -> None:
        if ev.phase == "downloading" and ev.total_bytes:
            pct = ev.percent
            print(f"  {pct:5.1f}%  ({ev.downloaded_bytes / 1e6:.1f} MB / {ev.total_bytes / 1e6:.1f} MB)")

    try:
        path = download(entry, progress=_cb)
        print(f"\nSaved to: {path}")
    except ManualDownloadRequired as exc:
        print(str(exc))
    except DownloadError as exc:
        print(f"Download failed: {exc}")


if __name__ == "__main__":
    main()
