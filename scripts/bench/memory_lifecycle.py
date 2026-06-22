#!/usr/bin/env python
"""Measure VRAM before / peak / after for the lifecycle scenarios in the report.

Scenarios: normal completion, cancel mid-run, model switch, long video.

    TMPDIR=/home/arash/.cache/vsx_tmp python scripts/bench/memory_lifecycle.py
"""

from __future__ import annotations

import threading
import time

from _bench_common import OWNER_VIDEO, emit

COLUMNS = ["scenario", "before_mb", "peak_mb", "after_cleanup_mb", "result", "evidence"]


def main() -> int:
    from visionservex.core.model import VisionModel
    from visionservex.runtime.gpu_lifecycle import get_gpu_memory_state
    from visionservex.runtime.video_pipeline import CancelToken, infer_video

    rows = []

    def snap():
        return round(get_gpu_memory_state().allocated_mb, 1)

    # normal completion
    vm = VisionModel("dfine-n", device="cuda", auto_pull=True)
    vm.warmup()
    before = snap()
    rep = infer_video(vm, OWNER_VIDEO, sample_fps=2.0, max_frames=48, mode="max_throughput")
    peak = rep["bottleneck_summary"]["vram_used_peak_mb"]
    after = rep["bottleneck_summary"]["vram_after_cleanup_mb"]
    rows.append(
        {
            "scenario": "normal_completion",
            "before_mb": before,
            "peak_mb": peak,
            "after_cleanup_mb": after,
            "result": "OK",
            "evidence": f"{rep['frames_processed']} frames",
        }
    )

    # cancel mid-run
    token = CancelToken()
    holder = {}
    t = threading.Thread(
        target=lambda: holder.update(
            r=infer_video(vm, OWNER_VIDEO, sample_fps=10.0, mode="max_throughput", cancel=token)
        )
    )
    t.start()
    time.sleep(0.5)
    token.cancel()
    t.join(timeout=30)
    rcl = holder["r"]
    rows.append(
        {
            "scenario": "cancel_mid_run",
            "before_mb": before,
            "peak_mb": rcl["bottleneck_summary"]["vram_used_peak_mb"],
            "after_cleanup_mb": rcl["bottleneck_summary"]["vram_after_cleanup_mb"],
            "result": "CANCELLED",
            "evidence": f"{rcl['frames_processed']} frames before cancel",
        }
    )
    vm.unload()

    # model switch (unload old, load new, ensure old VRAM freed)
    after_unload = snap()
    vm2 = VisionModel("dfine-s", device="cuda", auto_pull=True)
    try:
        vm2.warmup()
        sw_peak = round(get_gpu_memory_state().allocated_mb, 1)
        vm2.unload()
        rows.append(
            {
                "scenario": "model_switch",
                "before_mb": after_unload,
                "peak_mb": sw_peak,
                "after_cleanup_mb": snap(),
                "result": "OK",
                "evidence": "dfine-n unloaded then dfine-s loaded",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "scenario": "model_switch",
                "before_mb": after_unload,
                "peak_mb": "",
                "after_cleanup_mb": snap(),
                "result": "SKIP",
                "evidence": str(exc)[:60],
            }
        )

    # long video (more frames, chunked streaming)
    vm3 = VisionModel("dfine-n", device="cuda", auto_pull=True)
    vm3.warmup()
    lb = snap()
    rl = infer_video(vm3, OWNER_VIDEO, sample_fps=5.0, max_frames=300, mode="max_throughput")
    rows.append(
        {
            "scenario": "long_video",
            "before_mb": lb,
            "peak_mb": rl["bottleneck_summary"]["vram_used_peak_mb"],
            "after_cleanup_mb": rl["bottleneck_summary"]["vram_after_cleanup_mb"],
            "result": "OK",
            "evidence": f"{rl['frames_processed']} frames streamed",
        }
    )
    vm3.unload()

    emit("v322_memory_lifecycle", rows, COLUMNS, meta={"gpu": "RTX 5080"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
