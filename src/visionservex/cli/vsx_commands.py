"""v3.1 unified CLI groups: ``sam``, ``dino``, ``pipeline``, ``cv2-pro``.

Every command supports ``--explain`` (prints license/auth/schema/limitations/next
command). ``run`` executes commercial-safe runnable models; gated / non-commercial /
not-yet-released targets fail gracefully with the exact lawful next step (never a
token leak, never a fake result).
"""

from __future__ import annotations

import json

import typer

from visionservex.vsx import _DINO_FACTS, _SAM_FACTS, VSX, VSXError

sam_app = typer.Typer(
    help="SAM family: SAM 1/2/2.1/3, MobileSAM, EfficientSAM, MedSAM (status/run/explain)."
)
dino_app = typer.Typer(
    help="DINO family: DINOv2/v3 embeddings, GroundingDINO, DINO-X (status/embed/detect/explain)."
)
pipeline_app = typer.Typer(help="Composed text-to-mask pipelines (GroundingDINO+SAM).")
cv2_app = typer.Typer(help="CV2-Pro: professional commercial-safe OpenCV tools.")


def _echo(obj):
    typer.echo(json.dumps(obj, indent=2, default=str))


# ---------------- SAM ----------------
@sam_app.command("list")
def sam_list():
    """List SAM family model_ids with their honest state."""
    ids = sorted(set(" ".join(_SAM_FACTS.values()).split()) - {"edgesam"})
    _echo([{"model_id": m, "state": VSX.sam(m).status()} for m in ids])


@sam_app.command("status")
def sam_status(model_id: str):
    _echo(VSX.sam(model_id).explain())


@sam_app.command("run")
def sam_run(
    model_id: str,
    image: str,
    box: str = typer.Option(None, help="x1,y1,x2,y2"),
    out: str = typer.Option("runs/sam", help="output dir"),
    explain: bool = typer.Option(False, "--explain"),
):
    if explain:
        _echo(VSX.sam(model_id).explain())
        return
    box_coords = [int(v) for v in box.split(",")] if box else None
    try:
        res = VSX.sam(model_id).segment(image, box=box_coords)
        _echo({"status": "ok", "model_id": model_id, "out": out, "result_type": type(res).__name__})
    except VSXError as e:
        _echo(
            {
                "status": e.state,
                "model_id": model_id,
                "message": str(e),
                "next_command": e.next_command,
            }
        )
        raise typer.Exit(0) from None


@sam_app.command("video")
def sam_video(
    model_id: str, video: str, box: str = typer.Option(None), out: str = typer.Option("runs/video")
):
    try:
        VSX.sam(model_id).track(video, box=box)
    except VSXError as e:
        _echo({"status": e.state, "message": str(e), "next_command": e.next_command})


# ---------------- DINO ----------------
@dino_app.command("list")
def dino_list():
    ids = sorted(set(" ".join(_DINO_FACTS.values()).split()))
    _echo([{"model_id": m, "state": VSX.dino(m).status()} for m in ids])


@dino_app.command("status")
def dino_status(model_id: str):
    _echo(VSX.dino(model_id).explain())


@dino_app.command("embed")
def dino_embed(
    model_id: str,
    image: str,
    out: str = typer.Option("embedding.npy"),
    explain: bool = typer.Option(False, "--explain"),
):
    if explain:
        _echo(VSX.dino(model_id).explain())
        return
    try:
        VSX.dino(model_id).embed(image)
        _echo({"status": "ok", "model_id": model_id, "out": out})
    except VSXError as e:
        _echo({"status": e.state, "message": str(e), "next_command": e.next_command})


@dino_app.command("detect")
def dino_detect(
    model_id: str,
    image: str,
    text: str = typer.Option(...),
    out: str = typer.Option("boxes.json"),
    explain: bool = typer.Option(False, "--explain"),
):
    if explain:
        _echo(VSX.dino(model_id).explain())
        return
    try:
        VSX.dino(model_id).detect(image, text=text)
        _echo({"status": "ok", "model_id": model_id, "text": text, "out": out})
    except VSXError as e:
        _echo({"status": e.state, "message": str(e), "next_command": e.next_command})


@dino_app.command("api")
def dino_api(
    model_id: str, image: str, text: str = typer.Option(...), out: str = typer.Option("out/")
):
    e = VSX.dino(model_id).explain()
    _echo(
        {
            "status": e["state"],
            "model_id": model_id,
            "message": "API/BYOT model — set the API key env var; weights are never mirrored",
            "next_command": e["next_command"],
        }
    )


# ---------------- pipeline ----------------
@pipeline_app.command("list")
def pipeline_list(family: str = typer.Option("sam-dino", "--family")):
    pipes = [
        "grounding-dino-swin-t+sam-vit-h",
        "grounding-dino-swin-b+sam-vit-h",
        "grounding-dino-swin-t+sam2.1-hiera-small",
        "grounding-dino-swin-b+sam2.1-hiera-large",
        "grounding-dino-original-swin-t+sam2-hiera-small",
        "grounding-dino-original-swin-b+sam2-hiera-large",
        "grounding-dino-1.5+sam3-base",
        "grounding-dino-1.6+sam3-base",
        "dino-x-api+sam3-base",
        "dinov3-vitb16+sam2.1-hiera-small",
        "dinov3-vitb16+sam3-base",
    ]
    _echo([{"pipeline_id": p, "state": VSX.pipeline(p).status()} for p in pipes])


@pipeline_app.command("status")
def pipeline_status(pipeline_id: str):
    _echo(VSX.pipeline(pipeline_id).explain())


@pipeline_app.command("run")
def pipeline_run(
    pipeline_id: str,
    image: str,
    text: str = typer.Option(...),
    out: str = typer.Option("runs/text_to_mask"),
    explain: bool = typer.Option(False, "--explain"),
):
    if explain:
        _echo(VSX.pipeline(pipeline_id).explain())
        return
    try:
        VSX.pipeline(pipeline_id).run(image, text=text)
        _echo({"status": "ok", "pipeline_id": pipeline_id, "text": text, "out": out})
    except VSXError as e:
        _echo(
            {
                "status": e.state,
                "pipeline_id": pipeline_id,
                "message": str(e),
                "next_command": e.next_command,
            }
        )


# ---------------- cv2-pro ----------------
@cv2_app.command("list")
def cv2_list():
    from visionservex.cv2_pro import list_tools, tool_available

    _echo([{"tool": t, "available": tool_available(t)[0]} for t in list_tools()])


@cv2_app.command("run")
def cv2_run(
    tool: str,
    image: str,
    out: str = typer.Option("out.json"),
    onnx: str = typer.Option(None),
    mask: str = typer.Option(None),
    explain: bool = typer.Option(False, "--explain"),
):
    if explain:
        _echo(VSX.cv2(tool).explain())
        return
    params = {}
    if onnx:
        params["onnx"] = onnx
    try:
        res = VSX.cv2(tool).run(image, **params)
        from pathlib import Path

        Path(out).write_text(
            json.dumps({k: v for k, v in res.items() if k != "polygons"}, default=str, indent=2)
        )
        _echo(
            {
                "status": "ok",
                "tool": tool,
                "out": out,
                "kind": res.get("kind"),
                "latency_ms": res.get("latency_ms"),
            }
        )
    except Exception as e:
        _echo({"status": "error", "tool": tool, "message": str(e)})
