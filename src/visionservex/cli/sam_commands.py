# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""SAM CLI commands — VisionServeX v3.4.

Covers SAM v1, SAM 2, SAM 2.1, SAM 3/3.1, EfficientSAM, MobileSAM.

Commands:
  list         List SAM models from SOURCE_MANIFEST with their runnable status.
  status       Return structured JSON status for a given model ID.
  run          Run SAM segmentation on an image via VisionModel.
  export-onnx  Export the SAM mask-decoder to ONNX (Apache-2.0 eligible only).
  video        SAM2/2.1 video tracking on a directory of frames.

NEVER log HF tokens in full — use _redact() to show only the first 3 and last
2 characters (pattern mirrored from sam3_commands.py).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="sam", help="SAM-family commands (list/status/run/export-onnx/video).", no_args_is_help=True)
console = Console()

# SAM families covered by this CLI module.
_SAM_FAMILIES = {"sam", "sam2", "sam2.1", "sam3", "efficientsam", "mobilesam"}

# Models that are eligible for ONNX decoder export (Apache-2.0, local checkpoint).
# Mirrors onnx_export._SAM_ONNX_ELIGIBLE exactly.
_ONNX_ELIGIBLE = {"sam-vit-b", "sam-vit-l", "sam-vit-h", "mobilesam"}

# Checkpoint paths used for existence checks (mirrors onnx_export._SAM_ONNX_ELIGIBLE).
_CHECKPOINT_PATHS: dict[str, str] = {
    "sam-vit-b": "~/.cache/visionservex/sam/sam_vit_b_01ec64.pth",
    "sam-vit-l": "~/.cache/visionservex/sam/sam_vit_l_0b3195.pth",
    "sam-vit-h": "~/.cache/visionservex/sam/sam_vit_h_4b8939.pth",
    "mobilesam": "~/.cache/visionservex/mobilesam/mobile_sam.pt",
}

# SAM3 / SAM3.1 gated error code.
GATED_HF_AUTH_REQUIRED = "GATED_HF_AUTH_REQUIRED"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redact(token: str | None) -> str:
    """Show only the first 3 and last 2 chars of an HF token. Never log full token."""
    if not token:
        return ""
    if len(token) < 8:
        return "***"
    return f"{token[:3]}***{token[-2:]}"


def _is_sam3(model_id: str) -> bool:
    """Return True if model_id looks like a SAM3 or SAM3.1 gated model."""
    lid = model_id.lower()
    return "sam3" in lid or "sam-3" in lid or "sam3.1" in lid


def _checkpoint_exists(model_id: str) -> bool:
    """Return True if the known checkpoint path for model_id is present on disk."""
    raw = _CHECKPOINT_PATHS.get(model_id)
    if raw is None:
        return False
    return Path(raw).expanduser().exists()


def _sam_entries_from_manifest():
    """Return all SOURCE_MANIFEST entries whose family is in _SAM_FAMILIES."""
    from visionservex.model_zoo import SOURCE_MANIFEST
    return [e for e in SOURCE_MANIFEST.values() if e.family in _SAM_FAMILIES]


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command("list")
def list_cmd(
    json_: bool = typer.Option(False, "--json", help="Machine-readable JSON output."),
    explain: bool = typer.Option(False, "--explain", help="Print a one-paragraph explanation before the results."),
) -> None:
    """List SAM models from SOURCE_MANIFEST with their runnable status."""
    if explain:
        console.print(
            "The 'sam list' command queries the VisionServeX SOURCE_MANIFEST for every model "
            "whose family is in {sam, sam2, sam2.1, sam3, efficientsam, mobilesam}. For each "
            "entry it shows the model ID, family, current runnable status in this build, "
            "license, and the ONNX-eligibility flag. Models marked 'runnable' can be used "
            "directly with 'visionservex sam run'. Models that are not runnable include a "
            "blocker note derived from their manifest entry."
        )

    entries = _sam_entries_from_manifest()

    if json_:
        rows = []
        for e in entries:
            rows.append({
                "model_id": e.model_id,
                "family": e.family,
                "runnable": e.runnable_in_visionservex,
                "license": e.license,
                "onnx_eligible": e.model_id in _ONNX_ELIGIBLE,
                "status": "runnable" if e.runnable_in_visionservex else "not_runnable",
                "install_command": e.install_command,
            })
        print(json.dumps(rows, indent=2))
        return

    table = Table(title=f"SAM-family models ({len(entries)})", show_header=True)
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Family", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("License", no_wrap=True)
    table.add_column("ONNX", no_wrap=True)
    for e in entries:
        run_label = "[green]runnable[/green]" if e.runnable_in_visionservex else "[dim]not_runnable[/dim]"
        onnx_label = "[cyan]yes[/cyan]" if e.model_id in _ONNX_ELIGIBLE else "-"
        table.add_row(e.model_id, e.family, run_label, e.license, onnx_label)
    console.print(table)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@app.command("status")
def status_cmd(
    model_id: str = typer.Argument(..., help="SAM model ID (e.g. sam-vit-b, sam2-hiera-tiny, sam3-base)."),
    json_: bool = typer.Option(False, "--json", help="Machine-readable JSON output."),
    explain: bool = typer.Option(False, "--explain", help="Print a one-paragraph explanation."),
) -> None:
    """Return structured status for a SAM model: checkpoint, ONNX eligibility, auth, blocker."""
    if explain:
        console.print(
            "The 'sam status' command inspects a given model ID and returns a structured "
            "status snapshot. For SAM3/SAM3.1, it always reports auth_required=True and the "
            "GATED_HF_AUTH_REQUIRED code because those weights are gated on Hugging Face. "
            "For sam-vit-b and mobilesam it performs a real on-disk checkpoint check and "
            "reports onnx_eligible=True. For all other models it consults SOURCE_MANIFEST to "
            "determine whether the model is runnable, what the license is, and what the "
            "current blocker is (if any)."
        )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""

    # SAM3 / SAM3.1 are always gated — return immediately without manifest lookup.
    if _is_sam3(model_id):
        payload = {
            "model_id": model_id,
            "checkpoint_exists": False,
            "onnx_eligible": False,
            "license": "Apache-2.0 (code), gated HF access (weights)",
            "runnable": False,
            "auth_required": True,
            "blocker": GATED_HF_AUTH_REQUIRED,
            "fix": (
                "1) Visit https://huggingface.co/facebook/sam3 and accept access terms. "
                "2) Run: huggingface-cli login   (or set HF_TOKEN env var). "
                "3) Tokens are redacted in all VisionServeX logs — "
                f"current token: {_redact(hf_token) or '(none)'}."
            ),
        }
        if json_:
            print(json.dumps(payload, indent=2))
            return
        _print_status_table(payload)
        return

    # For all other SAM models, consult the manifest.
    from visionservex.model_zoo import get_model_source

    src = get_model_source(model_id)

    onnx_eligible = model_id in _ONNX_ELIGIBLE
    checkpoint_exists = _checkpoint_exists(model_id) if onnx_eligible else False

    if src is None:
        payload = {
            "model_id": model_id,
            "checkpoint_exists": checkpoint_exists,
            "onnx_eligible": onnx_eligible,
            "license": "unknown",
            "runnable": False,
            "auth_required": False,
            "blocker": "MODEL_NOT_FOUND",
            "fix": "Run 'visionservex sam list' to see available models.",
        }
        if json_:
            print(json.dumps(payload, indent=2))
            return
        _print_status_table(payload)
        raise typer.Exit(1)

    # Determine blocker and fix from the manifest entry.
    if src.runnable_in_visionservex:
        blocker = ""
        fix = ""
    elif src.known_blockers:
        blocker = src.known_blockers[0]
        fix = src.install_command
    else:
        blocker = "MODEL_NOT_RUNNABLE"
        fix = src.install_command

    payload = {
        "model_id": model_id,
        "checkpoint_exists": checkpoint_exists,
        "onnx_eligible": onnx_eligible,
        "license": src.license,
        "runnable": src.runnable_in_visionservex,
        "auth_required": src.access_status in {"gated", "hf_login", "api_token"},
        "blocker": blocker,
        "fix": fix,
    }

    if json_:
        print(json.dumps(payload, indent=2))
        return

    _print_status_table(payload)


def _print_status_table(payload: dict) -> None:
    table = Table(title=f"SAM status — {payload['model_id']}", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("model_id", payload["model_id"])
    table.add_row("checkpoint_exists", "yes" if payload["checkpoint_exists"] else "no")
    table.add_row("onnx_eligible", "yes" if payload["onnx_eligible"] else "no")
    table.add_row("license", payload["license"])
    table.add_row("runnable", "yes" if payload["runnable"] else "no")
    table.add_row("auth_required", "yes" if payload["auth_required"] else "no")
    table.add_row("blocker", f"[yellow]{payload['blocker']}[/yellow]" if payload["blocker"] else "—")
    table.add_row("fix", payload["fix"] or "—")
    console.print(table)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command("run")
def run_cmd(
    model_id: str = typer.Argument(..., help="SAM model ID to run."),
    image: Path = typer.Argument(..., help="Input image path."),
    box: Optional[str] = typer.Option(None, "--box", help="Box prompt as x1,y1,x2,y2."),
    out: Optional[Path] = typer.Option(None, "--out", help="Path to write result JSON."),
    json_: bool = typer.Option(False, "--json", help="Machine-readable JSON output."),
    explain: bool = typer.Option(False, "--explain", help="Print a one-paragraph explanation."),
) -> None:
    """Run SAM segmentation on an image via VisionModel and save result JSON."""
    if explain:
        console.print(
            "The 'sam run' command loads the specified SAM model using VisionModel(model_id) "
            "and calls model.predict(image) with an optional bounding-box prompt. The result "
            "is serialised to JSON and written to --out (or printed to stdout if --json is "
            "given). The command exits with code 3 if the image is missing or the box format "
            "is invalid, and code 4 if inference fails. SAM3/SAM3.1 models return a "
            "structured auth error rather than attempting inference."
        )

    # SAM3 gating — refuse inference.
    if _is_sam3(model_id):
        err = {
            "code": GATED_HF_AUTH_REQUIRED,
            "model_id": model_id,
            "status": "expected_blocker",
            "message": f"{model_id!r} is gated; VisionServeX will not attempt inference without HF auth.",
            "fix": "Run: visionservex sam status " + model_id + " for auth instructions.",
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(err, indent=2))
        if json_:
            print(json.dumps(err, indent=2))
        else:
            console.print(f"[yellow]{GATED_HF_AUTH_REQUIRED}[/yellow]: {err['message']}")
        raise typer.Exit(0)

    if not image.exists():
        err = {
            "code": "INPUT_NOT_FOUND",
            "status": "failed",
            "message": f"Image not found: {image}",
            "fix": f"Check path: {image}",
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(err, indent=2))
        if json_:
            print(json.dumps(err, indent=2))
        else:
            console.print(f"[red]INPUT_NOT_FOUND[/red]: {err['message']}")
        raise typer.Exit(3)

    boxes = None
    if box:
        try:
            parts = [float(x.strip()) for x in box.split(",")]
            if len(parts) != 4:
                raise ValueError(f"Expected x1,y1,x2,y2 but got {len(parts)} values.")
            boxes = [parts]
        except ValueError as exc:
            err = {
                "code": "INPUT_SCHEMA_ERROR",
                "status": "failed",
                "message": f"Invalid box format {box!r}: {exc}",
                "fix": "Use --box x1,y1,x2,y2 with four numeric values.",
            }
            if out:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(err, indent=2))
            if json_:
                print(json.dumps(err, indent=2))
            else:
                console.print(f"[red]INPUT_SCHEMA_ERROR[/red]: {err['message']}")
            raise typer.Exit(3)

    try:
        from PIL import Image as PILImage
        from visionservex import VisionModel

        pil_image = PILImage.open(image).convert("RGB")
        model = VisionModel(model_id)
        predict_kwargs: dict = {}
        if boxes:
            predict_kwargs["boxes"] = boxes
        result = model.predict(pil_image, **predict_kwargs)
        n_segments = len(getattr(result, "segments", []) or [])
        payload = {
            "model_id": model_id,
            "image": str(image),
            "box": box,
            "status": "ok",
            "code": "OK",
            "n_segments": n_segments,
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2))
        if json_:
            print(json.dumps(payload, indent=2))
        else:
            console.print(f"[green]OK[/green]: {model_id} — {n_segments} segments")
    except Exception as exc:
        err = {
            "code": "RUN_ERROR",
            "status": "failed",
            "model_id": model_id,
            "message": str(exc)[:300],
            "fix": "Check that the model is runnable: visionservex sam status " + model_id,
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(err, indent=2))
        if json_:
            print(json.dumps(err, indent=2))
        else:
            console.print(f"[red]RUN_ERROR[/red]: {err['message']}")
        raise typer.Exit(4)


# ---------------------------------------------------------------------------
# export-onnx
# ---------------------------------------------------------------------------


@app.command("export-onnx")
def export_onnx_cmd(
    model_id: str = typer.Argument(..., help="SAM model ID to export (must be ONNX-eligible)."),
    out: Path = typer.Option(Path("sam_decoder.onnx"), "--out", help="Output .onnx path."),
    json_: bool = typer.Option(False, "--json", help="Machine-readable JSON output."),
    explain: bool = typer.Option(False, "--explain", help="Print a one-paragraph explanation."),
) -> None:
    """Export the SAM mask-decoder to ONNX (Apache-2.0 eligible models only)."""
    if explain:
        console.print(
            "The 'sam export-onnx' command calls onnx_export.export_sam_decoder_onnx() to "
            "convert the SAM mask-decoder (the prompt-conditioned, latency-critical part of "
            "the pipeline) to ONNX format using the official SAM SamOnnxModel wrapper. Only "
            "Apache-2.0 commercial-safe checkpoints (sam-vit-b, mobilesam) are eligible. The "
            "checkpoint must already be present on disk; pull it first with "
            "'visionservex model pull <model_id>'. The resulting ONNX file can be run on CPU "
            "via onnxruntime or deployed to edge/browser/WebGPU targets."
        )

    if model_id not in _ONNX_ELIGIBLE:
        err = {
            "code": "NOT_ONNX_ELIGIBLE",
            "model_id": model_id,
            "status": "failed",
            "message": f"{model_id!r} is not ONNX-eligible. Eligible models: {sorted(_ONNX_ELIGIBLE)}.",
            "fix": "Use sam-vit-b or mobilesam.",
        }
        if json_:
            print(json.dumps(err, indent=2))
        else:
            console.print(f"[red]NOT_ONNX_ELIGIBLE[/red]: {err['message']}")
        raise typer.Exit(1)

    try:
        from visionservex.onnx_export import export_sam_decoder_onnx

        result = export_sam_decoder_onnx(model_id, out)
        if json_:
            print(json.dumps(result, indent=2))
        else:
            console.print(f"[green]ONNX export OK[/green]: {result['onnx_path']} ({result['size_mb']} MB)")
    except FileNotFoundError as exc:
        err = {
            "code": "CHECKPOINT_MISSING",
            "model_id": model_id,
            "status": "failed",
            "message": str(exc),
            "fix": f"visionservex model pull {model_id}",
        }
        if json_:
            print(json.dumps(err, indent=2))
        else:
            console.print(f"[red]CHECKPOINT_MISSING[/red]: {err['message']}")
        raise typer.Exit(2)
    except Exception as exc:
        err = {
            "code": "ONNX_EXPORT_ERROR",
            "model_id": model_id,
            "status": "failed",
            "message": str(exc)[:300],
            "fix": "Ensure torch, onnx, and the SAM/MobileSAM package are installed.",
        }
        if json_:
            print(json.dumps(err, indent=2))
        else:
            console.print(f"[red]ONNX_EXPORT_ERROR[/red]: {err['message']}")
        raise typer.Exit(4)


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------


@app.command("video")
def video_cmd(
    model_id: str = typer.Argument(..., help="SAM2/2.1 model ID for video tracking."),
    frame_dir: Path = typer.Argument(..., help="Directory containing video frames (PNG/JPEG)."),
    out: Optional[Path] = typer.Option(None, "--out", help="Path to write tracking result JSON."),
    box: Optional[str] = typer.Option(None, "--box", help="Box prompt for frame 0 as x1,y1,x2,y2."),
    json_: bool = typer.Option(False, "--json", help="Machine-readable JSON output."),
    explain: bool = typer.Option(False, "--explain", help="Print a one-paragraph explanation."),
) -> None:
    """Run SAM2 video tracking on a directory of frames. Returns module_missing if sam2 not installed."""
    if explain:
        console.print(
            "The 'sam video' command uses the SAM2 runtime (sam2_runtime.track_video) to "
            "perform object tracking across a directory of image frames. Frame 0 is prompted "
            "with an optional bounding box; the tracker propagates the segmentation mask "
            "through subsequent frames. The command returns a structured 'module_missing' "
            "error if the sam2/transformers package is not available, rather than raising an "
            "unhandled ImportError. This command is only valid for SAM2 and SAM2.1 models; "
            "SAM3/3.1 models return GATED_HF_AUTH_REQUIRED immediately."
        )

    # SAM3 gating.
    if _is_sam3(model_id):
        err = {
            "code": GATED_HF_AUTH_REQUIRED,
            "model_id": model_id,
            "status": "expected_blocker",
            "message": f"{model_id!r} is gated; video tracking is not available without HF auth.",
            "fix": "Run: visionservex sam status " + model_id,
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(err, indent=2))
        if json_:
            print(json.dumps(err, indent=2))
        else:
            console.print(f"[yellow]{GATED_HF_AUTH_REQUIRED}[/yellow]: {err['message']}")
        raise typer.Exit(0)

    # Must be a SAM2 / SAM2.1 family model.
    sam2_families = {"sam2", "sam2.1"}
    from visionservex.model_zoo import get_model_source
    src = get_model_source(model_id)
    if src is not None and src.family not in sam2_families:
        err = {
            "code": "NOT_SAM2_FAMILY",
            "model_id": model_id,
            "status": "failed",
            "message": f"{model_id!r} has family '{getattr(src, 'family', 'unknown')}'; video tracking requires a SAM2/SAM2.1 model.",
            "fix": "Use a sam2-hiera-* or sam2.1-hiera-* model ID.",
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(err, indent=2))
        if json_:
            print(json.dumps(err, indent=2))
        else:
            console.print(f"[red]NOT_SAM2_FAMILY[/red]: {err['message']}")
        raise typer.Exit(1)

    if not frame_dir.is_dir():
        err = {
            "code": "FRAME_DIR_NOT_FOUND",
            "status": "failed",
            "message": f"Frame directory not found: {frame_dir}",
            "fix": f"Check that the path exists and contains image files: {frame_dir}",
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(err, indent=2))
        if json_:
            print(json.dumps(err, indent=2))
        else:
            console.print(f"[red]FRAME_DIR_NOT_FOUND[/red]: {err['message']}")
        raise typer.Exit(3)

    # Check for transformers / sam2 availability before loading frames.
    try:
        import transformers  # noqa: F401
    except ImportError:
        err = {
            "code": "module_missing",
            "module": "transformers",
            "model_id": model_id,
            "status": "expected_blocker",
            "message": "transformers is not installed; SAM2 video tracking is unavailable.",
            "fix": "pip install 'visionservex[hf]'",
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(err, indent=2))
        if json_:
            print(json.dumps(err, indent=2))
        else:
            console.print(f"[yellow]module_missing[/yellow]: {err['message']}")
        raise typer.Exit(0)

    # Collect frames from the directory.
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    frame_paths = sorted(
        p for p in frame_dir.iterdir() if p.suffix.lower() in extensions
    )
    if not frame_paths:
        err = {
            "code": "NO_FRAMES_FOUND",
            "status": "failed",
            "message": f"No image files found in {frame_dir}",
            "fix": "Ensure the directory contains PNG/JPEG frames.",
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(err, indent=2))
        if json_:
            print(json.dumps(err, indent=2))
        else:
            console.print(f"[red]NO_FRAMES_FOUND[/red]: {err['message']}")
        raise typer.Exit(3)

    # Parse optional box.
    parsed_box = None
    if box:
        try:
            parts = [float(x.strip()) for x in box.split(",")]
            if len(parts) != 4:
                raise ValueError(f"Expected x1,y1,x2,y2 but got {len(parts)} values.")
            parsed_box = parts
        except ValueError as exc:
            err = {
                "code": "INPUT_SCHEMA_ERROR",
                "status": "failed",
                "message": f"Invalid box format {box!r}: {exc}",
                "fix": "Use --box x1,y1,x2,y2",
            }
            if out:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(err, indent=2))
            if json_:
                print(json.dumps(err, indent=2))
            else:
                console.print(f"[red]INPUT_SCHEMA_ERROR[/red]: {err['message']}")
            raise typer.Exit(3)

    try:
        from PIL import Image as PILImage
        from visionservex.sam2_runtime import track_video

        frames = [PILImage.open(p).convert("RGB") for p in frame_paths]
        result = track_video(model_id, frames, box=parsed_box)
        payload = {
            "model_id": model_id,
            "frame_dir": str(frame_dir),
            "n_frames": len(frames),
            "box": box,
            "status": "ok",
            "code": "OK",
            **result,
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2))
        if json_:
            print(json.dumps(payload, indent=2))
        else:
            console.print(f"[green]OK[/green]: tracked {len(frames)} frames with {model_id}")
    except ImportError as exc:
        err = {
            "code": "module_missing",
            "module": str(exc),
            "model_id": model_id,
            "status": "expected_blocker",
            "message": f"A required module is not installed: {exc}",
            "fix": "pip install 'visionservex[hf]'",
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(err, indent=2))
        if json_:
            print(json.dumps(err, indent=2))
        else:
            console.print(f"[yellow]module_missing[/yellow]: {err['message']}")
        raise typer.Exit(0)
    except Exception as exc:
        err = {
            "code": "VIDEO_TRACK_ERROR",
            "model_id": model_id,
            "status": "failed",
            "message": str(exc)[:300],
            "fix": "Check transformers version and that the SAM2 checkpoint is cached.",
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(err, indent=2))
        if json_:
            print(json.dumps(err, indent=2))
        else:
            console.print(f"[red]VIDEO_TRACK_ERROR[/red]: {err['message']}")
        raise typer.Exit(4)


__all__ = ["app"]
