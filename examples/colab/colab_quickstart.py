# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""VisionServeX Colab GPU Worker — quickstart script.

Run this as a Python script or paste cells from it into a Colab notebook.

It demonstrates:
  1. Environment check (Colab + GPU)
  2. Optional Drive-backed cache
  3. Starting the gateway with the colab-gpu-worker profile
  4. Running a sample inference

Notes:
  - This script is intended for use *inside* a Colab session, but the
    code paths degrade gracefully off-Colab.
  - The gateway is bound to 127.0.0.1 — to expose externally, use
    `visionservex colab tunnel-start` (requires auth and explicit ack).
  - No model weights or user data are committed to the repository.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def step(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    step("1. Environment check")
    subprocess.run([sys.executable, "-m", "visionservex.cli.main", "colab", "doctor"])

    step("2. Optional: configure cache directory")
    cache = os.environ.get(
        "VISIONSERVEX_CACHE_DIR",
        "/content/visionservex_cache" if Path("/content").exists() else None,
    )
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
        os.environ["VISIONSERVEX_CACHE_DIR"] = cache
        print(f"Cache dir: {cache}")
    else:
        print("No cache dir override (using default).")

    step("3. Start the gateway with the colab-gpu-worker profile")
    print("Run in a separate cell:")
    print("  !visionservex gateway start --profile colab-gpu-worker &")
    print("  !sleep 3 && curl -s http://127.0.0.1:8080/health")

    step("4. Try a sample inference from Python")
    print(
        """
from visionservex import VisionModel
from PIL import Image

img = Image.new('RGB', (640, 480), color=(120, 120, 200))
result = VisionModel('dfine-n', device='auto').predict(img)
print(result.summary())
"""
    )

    step("5. Optional: expose via Cloudflare Tunnel (auth required)")
    print(
        """
!visionservex colab token              # generate API key; copy it
%env VISIONSERVEX_AUTH__API_KEY=<paste>
%env VISIONSERVEX_AUTH__ENABLED=true

!visionservex colab tunnel-start --domain api.example.com \\
    --i-understand-this-is-public
"""
    )


if __name__ == "__main__":
    main()
