# SPDX-License-Identifier: Apache-2.0
"""v2.54.0: CUDA runtime probe and GPU capability detection."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field


@dataclass
class CudaProbeResult:
    torch_version: str = ""
    torch_cuda_version: str = ""
    cuda_available: bool = False
    device_name: str = ""
    compute_capability: list[int] = field(default_factory=list)
    driver_version: str = ""
    matmul_ok: bool = False
    matmul_ms: float | None = None
    cuda_fallback: bool = False
    cuda_fallback_reason: str = ""
    preferred_runtime: str = ""
    blackwell_sm120: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def probe_cuda() -> CudaProbeResult:
    """Run CUDA capability probe. Returns structured result."""
    result = CudaProbeResult()
    try:
        import torch

        result.torch_version = torch.__version__
        result.torch_cuda_version = torch.version.cuda or ""
        result.cuda_available = torch.cuda.is_available()

        if result.cuda_available:
            result.device_name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            result.compute_capability = list(cap)
            result.blackwell_sm120 = cap == (12, 0) or cap[0] >= 12

            # CUDA version check
            if result.torch_cuda_version:
                cuda_major = int(result.torch_cuda_version.split(".")[0])
                if result.blackwell_sm120 and cuda_major < 13:
                    result.cuda_fallback = True
                    result.cuda_fallback_reason = (
                        f"GPU is Blackwell (sm_120) but torch cuda build is cu{result.torch_cuda_version.replace('.', '')[:3]}. "
                        f"Requires torch built against cu130 or later."
                    )
                    result.preferred_runtime = "upgrade_to_cu130"

            # Run matmul kernel test
            if not result.cuda_fallback:
                try:
                    x = torch.randn(2048, 2048, device="cuda")
                    y = torch.randn(2048, 2048, device="cuda")
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    z = x @ y
                    torch.cuda.synchronize()
                    result.matmul_ms = round((time.perf_counter() - t0) * 1000, 1)
                    result.matmul_ok = True
                    _ = float(z.mean())
                except Exception as e:
                    result.matmul_ok = False
                    result.cuda_fallback = True
                    result.cuda_fallback_reason = f"CUDA kernel test failed: {str(e)[:200]}"
    except ImportError:
        result.cuda_fallback = True
        result.cuda_fallback_reason = "torch not importable"

    # nvidia-smi driver version
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        result.driver_version = r.stdout.strip()
    except Exception:
        result.driver_version = "unknown"

    if not result.cuda_fallback and result.blackwell_sm120:
        result.preferred_runtime = "pose_mmpose_cu130"

    return result
