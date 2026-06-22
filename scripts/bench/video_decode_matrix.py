#!/usr/bin/env python
"""Benchmark the worker video toolkit: ffprobe, faststart remux, transcode, decode.

TMPDIR=/home/arash/.cache/vsx_tmp python scripts/bench/video_decode_matrix.py
"""

from __future__ import annotations

import os
import tempfile
import time

from _bench_common import OWNER_VIDEO, emit

COLUMNS = ["operation", "detail", "elapsed_ms", "result", "evidence"]


def main() -> int:
    from visionservex.runtime.ffmpeg_tools import (
        detect_hwaccel,
        probe_video,
        remux_faststart,
        transcode_browser_h264,
    )
    from visionservex.runtime.video_pipeline import extract_frames_to_list

    rows = []
    hw = detect_hwaccel()
    rows.append(
        {
            "operation": "detect_hwaccel",
            "detail": "nvenc/nvdec",
            "elapsed_ms": "",
            "result": "OK",
            "evidence": f"nvenc={hw['nvenc']} nvdec={hw['nvdec_cuvid']}",
        }
    )

    t0 = time.perf_counter()
    p = probe_video(OWNER_VIDEO)
    rows.append(
        {
            "operation": "ffprobe",
            "detail": f"{p.video_codec}/{p.pix_fmt}/{p.width}x{p.height}",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            "result": "OK",
            "evidence": f"faststart={p.faststart} action={p.recommended_action}",
        }
    )

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "fast.mp4")
        r = remux_faststart(OWNER_VIDEO, out)
        rows.append(
            {
                "operation": "remux_faststart",
                "detail": "lossless -c copy",
                "elapsed_ms": r.get("elapsed_ms"),
                "result": "OK" if r["ok"] else "FAIL",
                "evidence": f"out_faststart={r.get('output_faststart')} size={r.get('output_size_bytes')}",
            }
        )

        out2 = os.path.join(td, "720p.mp4")
        r2 = transcode_browser_h264(OWNER_VIDEO, out2, preset="720p")
        rows.append(
            {
                "operation": "transcode_browser_h264",
                "detail": f"720p/{r2.get('encoder')}",
                "elapsed_ms": r2.get("elapsed_ms"),
                "result": "OK" if r2["ok"] else "FAIL",
                "evidence": f"nvenc={r2.get('nvenc_used')} size={r2.get('output_size_bytes')}",
            }
        )

    for sfps in (1.0, 5.0, 10.0):
        t0 = time.perf_counter()
        frames = extract_frames_to_list(OWNER_VIDEO, sample_fps=sfps, max_frames=120)
        dt = (time.perf_counter() - t0) * 1000
        rows.append(
            {
                "operation": "extract_frames",
                "detail": f"sample_fps={sfps}",
                "elapsed_ms": round(dt, 1),
                "result": "OK",
                "evidence": f"{len(frames)} frames, {round(len(frames) / (dt / 1000), 1)} fps decode",
            }
        )

    emit(
        "v322_video_decode_matrix",
        rows,
        COLUMNS,
        meta={"video": OWNER_VIDEO, "hwaccel": hw["nvenc_available"]},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
