# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Minimal Python HTTP client for the VisionServeX API.

Demonstrates upload + base64 + auth + job polling.
"""

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

import httpx


def client() -> httpx.Client:
    headers = {}
    api_key = os.environ.get("VISIONSERVEX_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return httpx.Client(
        base_url=os.environ.get("VSX_HOST", "http://127.0.0.1:8080"),
        headers=headers,
        timeout=60.0,
    )


def detect_multipart(image_path: Path, model_id: str) -> dict:
    with client() as c, image_path.open("rb") as fh:
        r = c.post(
            "/detect",
            data={"model_id": model_id},
            files={"image": (image_path.name, fh, "image/jpeg")},
        )
        r.raise_for_status()
        return r.json()


def open_vocab_b64(image_path: Path, model_id: str, prompts: list[str]) -> dict:
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    with client() as c:
        r = c.post(
            "/open-vocab/detect",
            json={"model_id": model_id, "image_b64": b64, "prompts": prompts},
        )
        r.raise_for_status()
        return r.json()


def poll_job(job_id: str, *, timeout_s: float = 600.0) -> dict:
    deadline = time.time() + timeout_s
    with client() as c:
        while time.time() < deadline:
            r = c.get(f"/jobs/{job_id}")
            r.raise_for_status()
            data = r.json()
            if data["status"] in {"completed", "failed", "cancelled"}:
                return data
            time.sleep(1.0)
    raise TimeoutError(f"job {job_id} did not finish in {timeout_s} s")


def main() -> int:
    img = Path(sys.argv[1] if len(sys.argv) >= 2 else "examples/images/street.jpg")
    if not img.exists():
        print(f"sample image not found: {img}")
        return 2
    out = detect_multipart(img, "mock-detect")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
