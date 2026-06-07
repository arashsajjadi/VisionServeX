# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Instance segmentation CLI (RF-DETR-Seg) — VisionServeX v3.7.

  visionservex segment-instances image.jpg --model rfdetr-seg-small --out out/ --explain

All six RF-DETR-Seg variants are Apache-2.0 / commercial-safe. The seg XL/2XL are
Apache-2.0 and do NOT require the PML-1.0 rfdetr_plus package.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="segment-instances",
                  help="RF-DETR instance segmentation (Apache-2.0).",
                  no_args_is_help=False, invoke_without_command=True)


@app.callback(invoke_without_command=True)
def main(
    image: Optional[str] = typer.Argument(None, help="Input image path."),
    model: str = typer.Option("rfdetr-seg-small", "--model", help="RF-DETR-Seg variant."),
    threshold: float = typer.Option(0.3, "--threshold"),
    out: Optional[str] = typer.Option(None, "--out", help="Output directory."),
    explain: bool = typer.Option(False, "--explain"),
    fmt: str = typer.Option("json", "--format"),
) -> None:
    """Run RF-DETR instance segmentation on an image."""
    from visionservex.rfdetr_seg_runtime import explain as _ex
    if explain or image is None:
        typer.echo(json.dumps(_ex(model), indent=2, default=str))
        return
    from PIL import Image
    from visionservex.rfdetr_seg_runtime import segment_instances
    try:
        res = segment_instances(model, Image.open(image).convert("RGB"), threshold=threshold)
        payload = {k: v for k, v in res.items() if k != "detections"}
        payload["status"] = "ok"
        if out:
            outd = Path(out); outd.mkdir(parents=True, exist_ok=True)
            (outd / f"{model}_instances.json").write_text(json.dumps(payload, indent=2, default=str))
            payload["artifact"] = str(outd / f"{model}_instances.json")
    except Exception as e:
        payload = {"model_id": model, "status": "expected_blocker",
                   "code": "RFDETR_SEG_RUNTIME_BLOCKER", "message": str(e),
                   "next_command": "pip install 'visionservex[rfdetr]'"}
    typer.echo(json.dumps(payload, indent=2, default=str))
