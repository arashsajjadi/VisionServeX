# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Interactive (click-based) segmentation CLI — VisionServeX v3.7.

  list      List interactive segmenters with license/state.
  status    Structured status for one model (+ --explain).
  run       Run interactive segmentation with positive/negative point prompts.

Named deep models (ritm/clickseg/simpleclick/focalclick) carry honest BYOT/legal
states; classic refiners (grabcut/watershed/...) run immediately, commercial-safe.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="interactive", help="Click-based interactive segmentation.", no_args_is_help=True
)
console = Console()

_DEEP = ["ritm", "clickseg", "simpleclick", "focalclick"]
_CLASSIC = ["grabcut", "watershed", "random-walker", "slic-graphcut"]


def _load_points(spec: str | None):
    """Accept a JSON file path or inline JSON list of [x,y] pairs."""
    if not spec:
        return []
    p = Path(spec)
    raw = p.read_text() if p.exists() else spec
    data = json.loads(raw)
    return [tuple(pt) for pt in data]


@app.command("list")
def cmd_list(fmt: str = typer.Option("table", "--format")) -> None:
    """List all interactive segmenters with license/state."""
    from visionservex.interactive_runtime import explain as _ex

    rows = [_ex(m) for m in _DEEP + _CLASSIC]
    if fmt == "json":
        typer.echo(json.dumps(rows, indent=2, default=str))
        return
    t = Table(title="Interactive Segmenters (v3.7)")
    for c in ("model_id", "state", "license", "commercial_safe"):
        t.add_column(c)
    for r in rows:
        t.add_row(
            r["model_id"],
            str(r.get("state")),
            str(r.get("license", ""))[:48],
            str(r.get("commercial_safe", "")),
        )
    console.print(t)


@app.command("status")
def cmd_status(
    model_id: str = typer.Argument(...),
    explain: bool = typer.Option(False, "--explain", help="Print full explain() dict."),
    out: str | None = typer.Option(None, "--out"),
) -> None:
    """Structured status for an interactive segmenter."""
    from visionservex.interactive_runtime import explain as _ex

    info = _ex(model_id)
    payload = (
        info
        if explain
        else {
            "model_id": model_id,
            "state": info.get("state"),
            "commercial_safe": info.get("commercial_safe"),
            "next_command": info.get("next_command"),
        }
    )
    if out:
        Path(out).write_text(json.dumps(payload, indent=2, default=str))
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command("run")
def cmd_run(
    model_id: str = typer.Argument(...),
    image: str = typer.Argument(...),
    positive_points: str | None = typer.Option(
        None, "--positive-points", help="JSON file or inline list of [x,y]."
    ),
    negative_points: str | None = typer.Option(
        None, "--negative-points", help="JSON file or inline list of [x,y]."
    ),
    out: str | None = typer.Option(None, "--out", help="Output directory."),
    fmt: str = typer.Option("json", "--format"),
    explain: bool = typer.Option(False, "--explain"),
) -> None:
    """Run interactive segmentation with point prompts."""
    from visionservex.interactive_runtime import explain as _ex
    from visionservex.interactive_runtime import run_interactive

    if explain:
        typer.echo(json.dumps(_ex(model_id), indent=2, default=str))
        return
    from PIL import Image

    pos = _load_points(positive_points)
    neg = _load_points(negative_points)
    try:
        res = run_interactive(
            model_id, Image.open(image).convert("RGB"), positive_points=pos, negative_points=neg
        )
    except Exception as e:
        res = {
            "model_id": model_id,
            "status": "expected_blocker",
            "code": "INTERACTIVE_RUNTIME_BLOCKER",
            "message": str(e),
        }
    # persist mask if produced
    payload = {k: v for k, v in res.items() if k != "mask"}
    if out and res.get("mask") is not None:
        import numpy as np
        from PIL import Image as _I

        outd = Path(out)
        outd.mkdir(parents=True, exist_ok=True)
        m = (np.asarray(res["mask"]) * 255).astype("uint8")
        _I.fromarray(m).save(outd / f"{model_id}_mask.png")
        payload["mask_png"] = str(outd / f"{model_id}_mask.png")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / f"{model_id}_result.json").write_text(
            json.dumps(payload, indent=2, default=str)
        )
    typer.echo(json.dumps(payload, indent=2, default=str))
