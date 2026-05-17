# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Agriculture domain commands — prompt-detect, prompt-segment, training templates.

Links:
- AgriCLIP: https://github.com/umair1221/AgriCLIP
- SCOLD: https://huggingface.co/enalis/scold
- D-FINE: https://github.com/Peterande/D-FINE
- RF-DETR: https://github.com/roboflow/rf-detr
- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="Agriculture domain — weed detection, crop segmentation, training templates.",
    no_args_is_help=True,
)
console = Console()


@app.command("doctor")
def doctor(
    fmt: str = typer.Option("text", "--format", help="Output format: text or json."),
    out: Path = typer.Option(None, "--out", help="Write JSON output to this path."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Check which agriculture-domain components are available."""
    import importlib

    checks = {
        "owlv2 (prompt-detect)": "visionservex.engines.owlv2",
        "grounding-dino (prompt-detect)": "visionservex.engines.grounding_dino",
        "rfdetr (weed-detect)": "visionservex.engines.rfdetr",
        "dfine (weed-detect)": "visionservex.engines.dfine",
        "siglip2 (embedder)": "visionservex.engines.dinov2",
    }
    rows = []
    for name, mod_path in checks.items():
        try:
            importlib.import_module(mod_path)
            rows.append({"component": name, "status": "available"})
        except ImportError:
            rows.append({"component": name, "status": "not_available"})
    payload = {"components": rows}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if json_ or fmt == "json":
        print(json.dumps(payload, indent=2))
        return
    table = Table(title="Agriculture components", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status", no_wrap=True)
    for r in rows:
        st = (
            "[green]available[/green]" if r["status"] == "available" else "[dim]not_available[/dim]"
        )
        table.add_row(r["component"], st)
    console.print(table)


@app.command("recommend")
def recommend(
    goal: str = typer.Option(
        ..., "--goal", help="e.g. weed-detection, crop-segmentation, disease-classification"
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Recommend a model pipeline for an agriculture task."""
    RECIPES: dict[str, dict] = {
        "weed-detection": {
            "recommended_detector": "owlv2-base-patch16",
            "alternative_detector": "grounding-dino-swin-b",
            "prompts": "weed, broadleaf weed, grass weed",
            "cli": "visionservex agriculture prompt-detect image.jpg --prompt 'weed'",
            "notes": "Use rfdetr-small for faster inference on CPU. Fine-tune on crop-weed dataset for best precision.",
        },
        "crop-segmentation": {
            "recommended_detector": "rfdetr-seg-medium",
            "alternative_detector": "grounding-dino-swin-b",
            "prompts": "crop row, plant",
            "cli": "visionservex agriculture prompt-segment image.jpg --prompt 'crop row'",
            "notes": "rfdetr-seg-medium is runnable. SAM2 can be added for mask refinement.",
        },
        "disease-classification": {
            "recommended_detector": "dinov2-base",
            "alternative_detector": "siglip2-base-patch16-224",
            "prompts": "leaf with yellow disease, healthy leaf, brown spots",
            "cli": "visionservex embed dinov2-base diseased_leaf.jpg --out /tmp/embedding.npy",
            "notes": "Use DINOv2 embeddings + nearest-neighbor classifier. No fine-tuning required for coarse detection.",
        },
    }
    g = goal.lower().replace(" ", "-").replace("_", "-")
    rec = RECIPES.get(g, RECIPES.get("weed-detection"))
    payload = {"goal": goal, "recipe": rec}
    if json_:
        print(json.dumps(payload, indent=2))
        return
    for k, v in (rec or {}).items():
        console.print(f"  [cyan]{k}:[/cyan] {v}")


@app.command("prompt-detect")
def prompt_detect(
    image: Path = typer.Argument(..., help="Image file."),
    prompt: str = typer.Option("weed", "--prompt"),
    detector: str = typer.Option("owlv2-base-patch16", "--detector"),
    threshold: float = typer.Option(0.15, "--threshold"),
    out: Path = typer.Option(None, "--out"),
    draw: Path = typer.Option(None, "--draw", help="Save annotated image to this path."),
    auto_pull: bool = typer.Option(False, "--auto-pull"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Run prompt-guided detection on an agriculture image."""
    if not image.exists():
        console.print(f"[red]Image not found:[/red] {image}")
        raise typer.Exit(2)
    from visionservex import VisionModel

    m = VisionModel(detector, auto_pull=auto_pull)
    from PIL import Image as PILImage

    img = PILImage.open(image).convert("RGB")
    result = m.predict(img, prompt=prompt, threshold=threshold)
    payload = result.to_dict()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if draw:
        try:
            from visionservex.visualization import annotate_image as _annotate

            annotated = _annotate(image, payload)
            draw.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(draw)
        except Exception as exc:
            console.print(f"[yellow]DRAW_FAILED: {exc}[/yellow]")
    if json_:
        print(json.dumps(payload, indent=2))
        return
    console.print(f"[bold]{result.summary()}[/bold]")
    for d in result.detections[:5]:
        console.print(f"  {d.label}: {d.score:.3f}")


@app.command("prompt-segment")
def prompt_segment(
    image: Path = typer.Argument(..., help="Image file."),
    prompt: str = typer.Option("weed", "--prompt"),
    detector: str = typer.Option("owlv2-base-patch16", "--detector"),
    out: Path = typer.Option(None, "--out"),
    draw: Path = typer.Option(None, "--draw", help="Save annotated image to this path."),
    auto_pull: bool = typer.Option(False, "--auto-pull"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Run prompt-guided detection + suggest adding SAM2 for mask refinement."""
    console.print(
        f"[yellow]Note:[/yellow] prompt-segment runs detection ({detector}) and returns boxes. "
        "For pixel-level masks, add --refine-with-sam2 when that flag lands."
    )
    prompt_detect(
        image=image,
        prompt=prompt,
        detector=detector,
        threshold=0.15,
        out=out,
        draw=draw,
        auto_pull=auto_pull,
        json_=json_,
    )


@app.command("recipe")
def recipe(
    name: str = typer.Argument(
        ..., help="Recipe name: crop-weed-detection, disease-classification, crop-segmentation"
    ),
    format: str = typer.Option("text", "--format", help="text or markdown"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Print a step-by-step recipe for an agriculture workflow."""
    RECIPES_LONG: dict[str, list[str]] = {
        "crop-weed-detection": [
            "1. Pull a prompt-capable open-vocab detector:",
            "   visionservex model pull owlv2-base-patch16",
            "2. Run detection on your field images:",
            "   visionservex agriculture prompt-detect field.jpg --prompt 'weed, broadleaf weed'",
            "3. For fine-grained adaptation, export a training template:",
            "   visionservex agriculture export-training-template --model rfdetr-small --out data_template/",
            "4. Label examples and fine-tune (bring your own YOLO-format dataset).",
        ],
        "disease-classification": [
            "1. Pull DINOv2-base for visual features:",
            "   visionservex model pull dinov2-base",
            "2. Build an embedding index of reference leaf images:",
            "   visionservex index dinov2-base reference_leaves/ --out indexes/leaves",
            "3. Query a new leaf image:",
            "   visionservex search dinov2-base unknown_leaf.jpg --index indexes/leaves --top-k 5",
        ],
        "crop-segmentation": [
            "1. Use rfdetr-seg-medium for instance segmentation:",
            "   visionservex model pull rfdetr-seg-medium",
            "   visionservex predict rfdetr-seg-medium field.jpg --json",
            "2. For SAM2-based mask refinement, see: visionservex domain-zoo agriculture",
        ],
    }
    steps = RECIPES_LONG.get(name)
    if not steps:
        console.print(f"[red]Unknown recipe {name!r}.[/red] Available: {list(RECIPES_LONG)}")
        raise typer.Exit(2)
    if json_:
        print(json.dumps({"recipe": name, "steps": steps}, indent=2))
        return
    if format == "markdown":
        console.print(f"## Agriculture recipe: {name}\n")
        for s in steps:
            console.print(s)
    else:
        for s in steps:
            console.print(s)


@app.command("export-training-template")
def export_training_template(
    model: str = typer.Option("rfdetr-small", "--model"),
    out: Path = typer.Option(..., "--out"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Export a YOLO-format training data template for agriculture fine-tuning."""

    out.mkdir(parents=True, exist_ok=True)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (out / "dataset.yaml").write_text(
        f"# YOLO-format template generated by VisionServeX agriculture export-training-template\n"
        f"# Model: {model}\n"
        "path: .  # dataset root\n"
        "train: images/train\n"
        "val:   images/val\n"
        "nc: 2\n"
        "names: ['crop', 'weed']\n"
        "\n"
        "# --- Instructions ---\n"
        "# 1. Add images to images/train/ and images/val/\n"
        "# 2. Add YOLO-format labels to labels/train/ and labels/val/\n"
        "#    Label format: <class_id> <cx> <cy> <w> <h>  (all normalized 0-1)\n"
        "# 3. Fine-tune with roboflow/rf-detr or any YOLO-compatible trainer\n"
    )
    payload = {"model": model, "template_dir": str(out), "format": "yolo_v8"}
    if json_:
        print(json.dumps(payload, indent=2))
        return
    console.print(f"[green]Template written to[/green] {out}")
    console.print(f"  Edit {out}/dataset.yaml and add your images + labels.")


__all__ = ["app"]
