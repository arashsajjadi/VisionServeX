#!/usr/bin/env python
"""Shared helpers for v3.22 benchmarks: JSON+Markdown emit, frame decode, NVML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT_DIR = Path("docs/audits/evidence")
OWNER_VIDEO = "/home/arash/Downloads/lv_0_20260617224920.mp4"


def decode_frames(path: str, n: int):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (n * 10)
    step = max(1, total // max(1, n))
    frames = []
    i = 0
    while len(frames) < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(fr[:, :, ::-1]).convert("RGB"))
        i += step
    cap.release()
    while frames and len(frames) < n:
        frames.append(frames[-1].copy())
    return frames


def emit(
    name: str, rows: list[dict[str, Any]], columns: list[str], meta: dict | None = None
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"benchmark": name, "meta": meta or {}, "rows": rows}
    (OUT_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2))
    md = [f"# Benchmark: {name}", ""]
    if meta:
        md += [f"- **{k}**: {v}" for k, v in meta.items()] + [""]
    md.append("| " + " | ".join(columns) + " |")
    md.append("| " + " | ".join("---" for _ in columns) + " |")
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    (OUT_DIR / f"{name}.md").write_text("\n".join(md) + "\n")
    print(f"wrote {OUT_DIR / name}.json and .md ({len(rows)} rows)")
