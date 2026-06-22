# SPDX-License-Identifier: Apache-2.0
"""v3.22.0 — LIVE GPU tests (D-FINE true batch, video infer, memory, cancel).

Gated: only runs when ``VSX_LIVE_GPU=1`` AND CUDA is available. These exercise
the real GPU so they are excluded from CI / the safe suite.

    VSX_LIVE_GPU=1 python tools/qa/runtests.py tests/live/test_v322_live_gpu.py -q
"""

from __future__ import annotations

import os

import pytest

LIVE = os.environ.get("VSX_LIVE_GPU") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="set VSX_LIVE_GPU=1 to run live GPU tests")

VIDEO = "/home/arash/Downloads/lv_0_20260617224920.mp4"


def _cuda_or_skip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch


def _frames(n: int):
    cv2 = pytest.importorskip("cv2")
    from PIL import Image

    if not os.path.exists(VIDEO):
        pytest.skip("owner video not present")
    cap = cv2.VideoCapture(VIDEO)
    out = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * 50)
        ok, fr = cap.read()
        if ok:
            out.append(Image.fromarray(fr[:, :, ::-1]).convert("RGB"))
    cap.release()
    return out


def test_dfine_true_forward_batch_one_call():
    _cuda_or_skip()
    from visionservex.core.model import VisionModel
    from visionservex.runtime.batch_infer import verify_true_forward_batch

    frames = _frames(8)
    vm = VisionModel("dfine-n", device="cuda", auto_pull=True)
    vm.warmup()
    try:
        v = verify_true_forward_batch(vm.engine, frames)
        assert v["forward_calls"] == 1, "8 images must be ONE forward (true batch)"
        assert v["is_true_forward_batch"] is True
        assert v["claim_matches_reality"] is True
    finally:
        vm.unload()


def test_dfine_batch_results_independent():
    _cuda_or_skip()
    from visionservex.core.model import VisionModel

    frames = _frames(6)
    vm = VisionModel("dfine-n", device="cuda", auto_pull=True)
    vm.warmup()
    try:
        res = vm.batch_predict(frames, threshold=0.5)
        assert len(res) == len(frames)
        assert all(r.metadata.get("batch_mode") == "true_tensor_batch" for r in res)
    finally:
        vm.unload()


def test_video_infer_memory_released():
    _cuda_or_skip()
    from visionservex.core.model import VisionModel
    from visionservex.runtime.gpu_lifecycle import get_gpu_memory_state
    from visionservex.runtime.video_pipeline import infer_video

    vm = VisionModel("dfine-n", device="cuda", auto_pull=True)
    vm.warmup()
    try:
        report = infer_video(vm, VIDEO, sample_fps=2.0, max_frames=24, mode="max_throughput")
        assert report["frames_processed"] > 0
        idxs = [f["frame_index"] for f in report["frames"]]
        assert len(set(idxs)) == len(idxs)
        # peak should exceed post-cleanup
        bs = report["bottleneck_summary"]
        assert bs["vram_used_peak_mb"] >= bs["vram_after_cleanup_mb"]
    finally:
        vm.unload()
        after = get_gpu_memory_state()
        assert after.allocated_mb < 200, "VRAM should be released after unload"


def test_video_infer_cancellation():
    _cuda_or_skip()
    import threading
    import time

    from visionservex.core.model import VisionModel
    from visionservex.runtime.video_pipeline import CancelToken, infer_video

    vm = VisionModel("dfine-n", device="cuda", auto_pull=True)
    vm.warmup()
    token = CancelToken()
    holder = {}
    try:
        t = threading.Thread(
            target=lambda: holder.update(
                r=infer_video(vm, VIDEO, sample_fps=10.0, mode="max_throughput", cancel=token)
            )
        )
        t.start()
        time.sleep(0.5)
        token.cancel()
        t.join(timeout=30)
        assert holder["r"]["cancelled"] is True
    finally:
        vm.unload()
