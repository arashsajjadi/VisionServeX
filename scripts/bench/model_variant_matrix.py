#!/usr/bin/env python
"""Benchmark model variants across the batch ladder via the PUBLIC predict_batch API.

Proves true-batch-vs-loop honestly (forward-call verified) and records the
throughput / VRAM / GPU-util / per-stage-timing curve.

    TMPDIR=/home/arash/.cache/vsx_tmp python scripts/bench/model_variant_matrix.py dfine-n dfine-s
"""

from __future__ import annotations

import sys

from _bench_common import OWNER_VIDEO, decode_frames, emit

LADDER = [1, 2, 4, 8, 16]
COLUMNS = [
    "model",
    "variant",
    "task",
    "batch_mode",
    "true_batch",
    "microbatch",
    "frames",
    "resolution",
    "gpu_avg",
    "gpu_peak",
    "vram_peak_mb",
    "forward_ms",
    "preprocess_ms",
    "postprocess_ms",
    "throughput_fps",
    "detections_per_frame",
    "failure_reason",
]


def main() -> int:
    models = sys.argv[1:] or ["dfine-n"]
    from visionservex.core.model import VisionModel
    from visionservex.runtime.batch_infer import run_batch_with_telemetry, verify_true_forward_batch

    frames = decode_frames(OWNER_VIDEO, max(LADDER))
    res = f"{frames[0].size[0]}x{frames[0].size[1]}"
    rows = []
    for mid in models:
        try:
            vm = VisionModel(mid, device="cuda", auto_pull=True)
            vm.warmup()
        except Exception as exc:
            rows.append({"model": mid, "failure_reason": f"load: {str(exc)[:80]}"})
            continue
        verify = verify_true_forward_batch(vm.engine, frames[:8])
        for bs in LADDER:
            try:
                results, tel = run_batch_with_telemetry(vm, frames[:bs], threshold=0.3)
                dets = sum(
                    len(getattr(r, "detections", []) or getattr(r, "segments", [])) for r in results
                ) / max(1, len(results))
                rows.append(
                    {
                        "model": mid,
                        "variant": mid,
                        "task": vm.entry.task,
                        "batch_mode": tel.batch_mode,
                        "true_batch": verify["is_true_forward_batch"],
                        "microbatch": bs,
                        "frames": bs,
                        "resolution": res,
                        "gpu_avg": tel.gpu_util_avg,
                        "gpu_peak": tel.gpu_util_peak,
                        "vram_peak_mb": tel.vram_used_peak_mb,
                        "forward_ms": tel.forward_ms,
                        "preprocess_ms": tel.preprocess_ms,
                        "postprocess_ms": tel.postprocess_ms,
                        "throughput_fps": round(bs / (tel.total_ms / 1000.0), 1)
                        if tel.total_ms
                        else 0,
                        "detections_per_frame": round(dets, 1),
                        "failure_reason": "",
                    }
                )
                print(
                    f"  {mid} bs={bs} mode={tel.batch_mode} fwd={tel.forward_ms}ms vram={tel.vram_used_peak_mb}MB"
                )
            except Exception as exc:
                rows.append({"model": mid, "microbatch": bs, "failure_reason": str(exc)[:80]})
        vm.unload()
    emit(
        "v322_model_variant_matrix",
        rows,
        COLUMNS,
        meta={"gpu": "RTX 5080", "video": OWNER_VIDEO, "ladder": LADDER},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
