#!/usr/bin/env python
"""Benchmark end-to-end video inference (decode+batch+schedule) over the owner video.

TMPDIR=/home/arash/.cache/vsx_tmp python scripts/bench/video_batch_matrix.py dfine-n
"""

from __future__ import annotations

import sys

from _bench_common import OWNER_VIDEO, emit

COLUMNS = [
    "model",
    "mode",
    "sample_fps",
    "frames",
    "waves",
    "trajectory",
    "throughput_fps",
    "gpu_avg",
    "vram_peak_mb",
    "vram_after_mb",
    "bottleneck",
    "detections_per_frame",
]


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "dfine-n"
    from visionservex.core.model import VisionModel
    from visionservex.runtime.video_pipeline import infer_video

    vm = VisionModel(model, device="cuda", auto_pull=True)
    vm.warmup()
    rows = []
    for mode in ("balanced", "max_throughput", "low_latency"):
        for sfps in (1.0, 2.0, 5.0):
            rep = infer_video(
                vm, OWNER_VIDEO, sample_fps=sfps, max_frames=60, mode=mode, threshold=0.3
            )
            bs = rep["bottleneck_summary"]
            dets = sum(f["n_objects"] for f in rep["frames"]) / max(1, len(rep["frames"]))
            last = rep["scheduler_decisions"][-1] if rep["scheduler_decisions"] else {}
            rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "sample_fps": sfps,
                    "frames": rep["frames_processed"],
                    "waves": rep["waves"],
                    "trajectory": rep["batch_trajectory"],
                    "throughput_fps": rep["throughput_fps"],
                    "gpu_avg": bs["gpu_util_avg"],
                    "vram_peak_mb": bs["vram_used_peak_mb"],
                    "vram_after_mb": bs["vram_after_cleanup_mb"],
                    "bottleneck": last.get("bottleneck", ""),
                    "detections_per_frame": round(dets, 1),
                }
            )
            print(
                f"  {mode} sfps={sfps} fps={rep['throughput_fps']} traj={rep['batch_trajectory']}"
            )
    vm.unload()
    emit("v322_video_batch_matrix", rows, COLUMNS, meta={"gpu": "RTX 5080", "video": OWNER_VIDEO})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
