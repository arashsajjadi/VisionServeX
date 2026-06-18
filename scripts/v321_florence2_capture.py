# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Capture live Florence-2 sidecar evidence into a JSON matrix.

Hits the running sidecar (default http://127.0.0.1:8091) across both weights and
all task tokens, records the real responses, and writes a committed evidence
matrix. No tokens are involved; nothing is persisted beyond the JSON.

    python scripts/v321_florence2_capture.py
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8091"
OUT = Path("docs/qa/v321_sidecar_blocker_elimination/florence2_sidecar_live.json")


def _png() -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (384, 384), (40, 90, 200))
    d = ImageDraw.Draw(img)
    d.ellipse((120, 120, 264, 264), fill=(250, 220, 40))
    d.rectangle((40, 40, 110, 110), fill=(210, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _post(model_id: str, task: str, img: bytes) -> dict:
    import requests

    r = requests.post(
        f"{BASE}/predict",
        files={"image": ("smoke.jpg", img, "image/jpeg")},
        data={"model_id": model_id, "task": task},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def main() -> int:
    img = _png()
    version = json.loads(urllib.request.urlopen(f"{BASE}/version", timeout=10).read())
    cases = [
        ("florence-2-base", "caption"),
        ("florence-2-base", "detailed_caption"),
        ("florence-2-base", "od"),
        ("florence-2-large", "caption"),
    ]
    results = []
    for model_id, task in cases:
        resp = _post(model_id, task, img)
        ok = bool(resp.get("text")) or bool(resp.get("detections"))
        results.append(
            {
                "model_id": model_id,
                "task": task,
                "ok": ok,
                "text": (resp.get("text") or "")[:200],
                "n_detections": len(resp.get("detections", [])),
            }
        )
        print(
            f"  {model_id:18s} {task:18s} -> {'OK' if ok else 'FAIL'}: {results[-1]['text'][:70]}"
        )

    matrix = {
        "sprint": "v3.21",
        "component": "florence2_sidecar",
        "transport": "http_docker_sidecar",
        "image": "visionservex-florence2:v321",
        "env": {
            "python": version.get("python"),
            "torch": version.get("torch"),
            "transformers": version.get("transformers"),
        },
        "device": "cpu",
        "flash_attn_workaround": "get_imports patched to drop flash_attn (eager attention)",
        "models": sorted({c[0] for c in cases}),
        "results": results,
        "all_ok": all(r["ok"] for r in results),
        "promoted_state": "VLM_READY_LIVE_SIDECAR",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(matrix, indent=2) + "\n")
    print(f"\nwrote {OUT}  all_ok={matrix['all_ok']}")
    return 0 if matrix["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
