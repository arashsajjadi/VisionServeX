# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Medical imaging commands — opt-in optional workflows.

VisionServeX does NOT make diagnostic claims. These commands are research /
education only. The package never auto-installs medical dependencies (some
require GPU + a multi-GB checkpoint pull from public hosts).

Supported entry points in v1.9.0:
- TotalSegmentator (CT/MR multi-organ segmentation) — sidecar
- MedSAM HF (promptable medical image segmentation) — sidecar
- MONAI bundle validation — sidecar

Reference:
- TotalSegmentator: https://github.com/wasserth/TotalSegmentator
- nnU-Net:           https://github.com/MIC-DKFZ/nnUNet
- MedSAM:            https://github.com/bowang-lab/MedSAM
- MedSAM HF:         https://huggingface.co/wanglab/medsam-vit-base
- MedSAM2:           https://github.com/bowang-lab/MedSAM2
- SAM-Med2D:         https://github.com/OpenGVLab/SAM-Med2D
- MONAI:             https://github.com/Project-MONAI/MONAI
- MONAI Model Zoo:   https://monai.io/model-zoo.html
"""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Medical imaging optional workflows.", no_args_is_help=True)
console = Console()


DISCLAIMER = (
    "VisionServeX medical commands are RESEARCH AND EDUCATION ONLY. "
    "They do not provide medical diagnosis, treatment recommendation, or any "
    "clinical guidance. Do not use the outputs for patient care."
)


@dataclass
class MedicalError:
    code: str
    message: str
    fix: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MedicalModelInfo:
    model_id: str
    description: str
    upstream: str
    install: tuple[str, ...]
    required_modules: tuple[str, ...]
    structured_error_code: str
    license_note: str
    license_tier: str = "optional_extra"
    runtime_status: str = "optional_extra"
    checkpoint_status: str = "verified"
    checkpoint_url: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


MEDICAL_MODELS: dict[str, MedicalModelInfo] = {
    "totalsegmentator": MedicalModelInfo(
        model_id="totalsegmentator",
        description="TotalSegmentator core total — multi-organ CT/MR segmentation (Apache).",
        upstream="https://github.com/wasserth/TotalSegmentator",
        install=(
            "pip install TotalSegmentator",
            "# requires nibabel + nnunetv2 + torch; GPU strongly recommended",
        ),
        required_modules=("totalsegmentator",),
        structured_error_code="TOTALSEGMENTATOR_REQUIRED",
        license_note=("Apache-2.0 core total model. Re-verify license before public install path."),
        license_tier="optional_extra",
        runtime_status="optional_extra",
    ),
    "totalsegmentator-tissue": MedicalModelInfo(
        model_id="totalsegmentator-tissue",
        description=(
            "TotalSegmentator tissue / body-stats — proprietary commercial-license subtask."
        ),
        upstream="https://github.com/wasserth/TotalSegmentator",
        install=(
            "pip install TotalSegmentator",
            "# Requires a commercial license key — totalseg_set_license <KEY>",
        ),
        required_modules=("totalsegmentator",),
        structured_error_code="TOTALSEGMENTATOR_LICENSE_REQUIRED",
        license_note=(
            "Non-commercial / proprietary. Requires explicit license key — NOT auto-runnable."
        ),
        license_tier="non_core_license_optional",
        runtime_status="non_core_license_optional",
    ),
    "medsam": MedicalModelInfo(
        model_id="medsam",
        description="MedSAM (HF wanglab/medsam-vit-base) — promptable medical segmentation.",
        upstream="https://github.com/bowang-lab/MedSAM",
        install=(
            "pip install 'visionservex[hf]'",
            "# weights from https://huggingface.co/wanglab/medsam-vit-base",
        ),
        required_modules=("transformers",),
        structured_error_code="MEDICAL_EXTRA_REQUIRED",
        license_note="Apache-2.0. Research-grade; not validated for clinical decisions.",
    ),
    "medsam2": MedicalModelInfo(
        model_id="medsam2",
        description="MedSAM2 — video/volumetric promptable segmentation.",
        upstream="https://github.com/bowang-lab/MedSAM2",
        install=(
            "git clone https://github.com/bowang-lab/MedSAM2.git",
            "cd MedSAM2 && pip install -e .",
        ),
        required_modules=("medsam2",),
        structured_error_code="MEDSAM2_CHECKPOINT_UNVERIFIED",
        license_note=(
            "Apache-2.0 code / NON-COMMERCIAL weights (medical dataset provenance). "
            "Research-only expert sidecar — NOT commercial-safe, not for diagnosis."
        ),
        license_tier="expert_sidecar",
        runtime_status="expert_sidecar",
        checkpoint_status="unverified",
    ),
    "sam-med2d": MedicalModelInfo(
        model_id="sam-med2d",
        description="SAM-Med2D — SAM adapter for 2D medical images.",
        upstream="https://github.com/OpenGVLab/SAM-Med2D",
        install=(
            "git clone https://github.com/OpenGVLab/SAM-Med2D.git",
            "cd SAM-Med2D && pip install -e .",
        ),
        required_modules=("sam_med2d",),
        structured_error_code="MEDICAL_EXTRA_REQUIRED",
        license_note="Apache-2.0.",
    ),
    "nnunet-v2": MedicalModelInfo(
        model_id="nnunet-v2",
        description="nnU-Net v2 — self-configuring medical image segmentation.",
        upstream="https://github.com/MIC-DKFZ/nnUNet",
        install=("pip install nnunetv2",),
        required_modules=("nnunetv2",),
        structured_error_code="NNUNET_REQUIRED",
        license_note=(
            "Apache-2.0 framework. Generic universal pretrained weights are NOT guaranteed."
        ),
        license_tier="expert_sidecar",
        runtime_status="expert_sidecar",
        checkpoint_status="dataset_specific",
    ),
    "monai-bundles": MedicalModelInfo(
        model_id="monai-bundles",
        description="MONAI Model Zoo — curated medical model bundles.",
        upstream="https://monai.io/model-zoo.html",
        install=("pip install monai",),
        required_modules=("monai",),
        structured_error_code="MEDICAL_EXTRA_REQUIRED",
        license_note="Apache-2.0 framework. Bundles have individual licenses.",
    ),
    "auto3dseg": MedicalModelInfo(
        model_id="auto3dseg",
        description="MONAI Auto3DSeg — automated 3D segmentation pipelines.",
        upstream="https://docs.monai.io/en/stable/apps.html",
        install=("pip install monai",),
        required_modules=("monai",),
        structured_error_code="MEDICAL_EXTRA_REQUIRED",
        license_note="Apache-2.0.",
    ),
}


def _module_present(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def _missing(info: MedicalModelInfo) -> list[str]:
    return [m for m in info.required_modules if not _module_present(m)]


def _nifti_required(path: Path) -> MedicalError | None:
    """Check if NIfTI input parsing dependency (nibabel) is present when needed."""
    if path.suffix.lower() not in {".nii", ".gz"}:
        return None
    if _module_present("nibabel"):
        return None
    return MedicalError(
        code="NIFTI_IO_REQUIRED",
        message="NIfTI volume input requires the nibabel package",
        fix="pip install nibabel",
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("recommend")
def recommend(
    goal: str = typer.Option(
        ...,
        "--goal",
        help="Goal phrase, e.g. ct-segmentation, prompt-segment, organ-classification.",
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Recommend a medical model for a goal. Output is research guidance only."""
    g = goal.lower().strip()
    if "ct" in g or "organ" in g:
        rec = "totalsegmentator"
    elif "prompt" in g or "click" in g or "interactive" in g:
        rec = "medsam"
    elif "volume" in g or "3d" in g or "video" in g:
        rec = "medsam2"
    elif "monai" in g:
        rec = "monai-bundles"
    else:
        rec = "medsam"
    info = MEDICAL_MODELS[rec]
    payload = {
        "goal": goal,
        "recommendation": rec,
        "model_info": info.to_dict(),
        "disclaimer": DISCLAIMER,
    }
    if json_:
        print(json.dumps(payload, indent=2))
        return
    console.print(f"[yellow]Disclaimer:[/yellow] {DISCLAIMER}\n")
    console.print(f"[bold]Goal:[/bold] {goal}")
    console.print(f"[bold]Recommendation:[/bold] {rec}")
    console.print(f"  upstream: {info.upstream}")
    console.print(f"  license:  {info.license_note}")


@app.command("validate")
def validate(
    model_id: str = typer.Argument(..., help=f"Medical model: {', '.join(MEDICAL_MODELS)}"),
    fmt: str = typer.Option("text", "--format", help="Output format: text or json."),
    out: Path | None = typer.Option(None, "--out", help="Write structured JSON to this path."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Verify whether a medical model's dependencies are available."""
    if model_id not in MEDICAL_MODELS:
        err_payload = {
            "model_id": model_id,
            "status": "failed",
            "code": "UNKNOWN_MEDICAL_MODEL",
            "message": f"Unknown medical model {model_id!r}. Options: {list(MEDICAL_MODELS)}",
            "install_command": "",
            "docs": "",
            "warnings": [],
            "errors": [f"Unknown medical model {model_id!r}"],
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(err_payload, indent=2))
        if json_ or fmt == "json":
            print(json.dumps(err_payload, indent=2))
        else:
            console.print(
                f"[red]Unknown medical model {model_id!r}.[/red] Options: {list(MEDICAL_MODELS)}"
            )
        raise typer.Exit(2)
    info = MEDICAL_MODELS[model_id]
    missing = _missing(info)
    status = "ok" if not missing else "expected_blocker"
    payload = {
        "model_id": model_id,
        "status": status,
        "code": "" if not missing else (info.structured_error_code or "DEPENDENCY_REQUIRED"),
        "message": "" if not missing else f"Missing: {missing}",
        "install_command": "; ".join(info.install) if missing else "",
        "docs": info.upstream,
        "warnings": [],
        "errors": [],
        # extended fields
        "installed": not missing,
        "required_modules": list(info.required_modules),
        "missing_modules": missing,
        "structured_error_code": None if not missing else info.structured_error_code,
        "install_commands": list(info.install),
        "upstream": info.upstream,
        "license_note": info.license_note,
        "disclaimer": DISCLAIMER,
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if json_ or fmt == "json":
        print(json.dumps(payload, indent=2))
        return
    if missing:
        console.print(f"[yellow]{info.structured_error_code}[/yellow] — missing: {missing}")
        for c in info.install:
            console.print(f"  [cyan]{c}[/cyan]")
    else:
        console.print(f"[green]{model_id} dependencies are installed.[/green]")


@app.command("segment")
def segment(
    model_id: str = typer.Argument(..., help="Medical segmentation model id."),
    input: Path = typer.Argument(..., help="Input image or NIfTI volume."),
    box: list[str] = typer.Option(
        default=[],
        help="Box prompt 'x1,y1,x2,y2' (repeat for multiple boxes).",
    ),
    point: str = typer.Option("", "--point", help="Point prompt 'x,y' for MedSAM-style models."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    draw: Path = typer.Option(None, "--draw", help="Save mask overlay image to this path."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Run medical segmentation. Returns structured errors if deps missing."""
    if model_id not in MEDICAL_MODELS:
        console.print(f"[red]Unknown medical model {model_id!r}.[/red]")
        raise typer.Exit(2)
    info = MEDICAL_MODELS[model_id]

    # Check input
    if not input.exists():
        err = MedicalError(
            code="INPUT_NOT_FOUND", message=f"Input not found: {input}", fix=f"Check path: {input}"
        )
        _emit_err(err, json_)
        raise typer.Exit(3)

    # Check NIfTI parser dep when needed
    nifti_err = _nifti_required(input)
    if nifti_err is not None:
        _emit_err(nifti_err, json_)
        raise typer.Exit(3)

    # Check module deps
    missing = _missing(info)
    if missing:
        err = MedicalError(
            code=info.structured_error_code,
            message=f"Missing modules: {missing}",
            fix=" && ".join(info.install),
        )
        _emit_err(err, json_)
        raise typer.Exit(3)

    # MedSAM real inference via SAM HF engine
    if model_id == "medsam":
        _segment_medsam(
            input=input,
            boxes_raw=box,
            out=out,
            draw=draw,
            json_=json_,
        )
        return

    # Honest delegation — v1.9.0 does not duplicate upstream segmentation engines.
    out.mkdir(parents=True, exist_ok=True)
    next_step_lookup = {
        "totalsegmentator": f"TotalSegmentator -i {input} -o {out}",
        "medsam2": (
            "# MedSAM2 is a research-only expert sidecar (non-commercial weights, "
            "not commercial-safe). Native predictor/checkpoint not wired in core. "
            f"See {info.upstream} and engines/medsam2_sidecar.py."
        ),
        "sam-med2d": f"# See {info.upstream}/blob/main/README.md",
        "nnunet-v2": f"nnUNetv2_predict -i {input} -o {out} -d <DATASET_ID> -c 3d_fullres",
        "monai-bundles": "monai-bundle list ; then load with monai.bundle.load",
        "auto3dseg": f"# python -m monai.apps.auto3dseg AutoRunner run --input {input}",
    }
    delegation = next_step_lookup.get(model_id, "see upstream README")
    payload = {
        "model_id": model_id,
        "input": str(input),
        "out": str(out),
        "boxes": list(box) or None,
        "point": point or None,
        "status": "delegate_to_upstream",
        "next_step": delegation,
        "disclaimer": DISCLAIMER,
    }
    if json_:
        print(json.dumps(payload, indent=2))
        return
    console.print(f"[bold]{model_id}[/bold] — VisionServeX delegates to upstream:")
    console.print(f"  [cyan]{delegation}[/cyan]")


@app.command("doctor")
def doctor(json_: bool = typer.Option(False, "--json")) -> None:
    """Check medical dependency status."""
    results = []
    for model_id, info in MEDICAL_MODELS.items():
        missing = _missing(info)
        entry: dict = {
            "model_id": model_id,
            "installed": not missing,
            "required_modules": list(info.required_modules),
            "missing_modules": missing,
            "install_commands": list(info.install),
        }
        if missing:
            entry["code"] = info.structured_error_code
            entry["fix"] = " && ".join(info.install)
        results.append(entry)

    if json_:
        print(json.dumps(results, indent=2))
        return

    table = Table(title="Medical dependency status", show_header=True)
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Installed", no_wrap=True)
    table.add_column("Missing modules")
    table.add_column("Install command")
    for entry in results:
        installed_label = "[green]yes[/green]" if entry["installed"] else "[red]no[/red]"
        missing_label = ", ".join(entry["missing_modules"]) if entry["missing_modules"] else "-"
        install_label = entry["install_commands"][0] if entry["install_commands"] else "-"
        table.add_row(entry["model_id"], installed_label, missing_label, install_label)
    console.print(table)


@app.command("list")
def list_models(json_: bool = typer.Option(False, "--json")) -> None:
    """List supported medical models."""
    rows = []
    for k, info in MEDICAL_MODELS.items():
        rows.append(
            {
                "id": k,
                "installed": not _missing(info),
                "description": info.description,
                "upstream": info.upstream,
                "license_note": info.license_note,
            }
        )
    if json_:
        print(json.dumps(rows, indent=2))
        return
    console.print(f"[yellow]Disclaimer:[/yellow] {DISCLAIMER}\n")
    table = Table(title="Supported medical models", show_header=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Installed", no_wrap=True)
    table.add_column("Description")
    for r in rows:
        installed = "[green]yes[/green]" if r["installed"] else "[dim]no[/dim]"
        table.add_row(r["id"], installed, r["description"])
    console.print(table)


def _segment_medsam(
    *,
    input: Path,
    boxes_raw: list[str],
    out: Path,
    draw: Path | None = None,
    json_: bool,
) -> None:
    """Run MedSAM inference using the SAM HF engine. Saves mask PNG + metadata JSON."""
    import json as json_mod

    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)

    # Parse all box prompts
    parsed_boxes: list[list[float]] = []
    for idx, raw in enumerate(boxes_raw):
        try:
            parts = [float(x.strip()) for x in raw.split(",")]
            if len(parts) != 4:
                raise ValueError(f"Expected x1,y1,x2,y2 got {len(parts)} values")
            parsed_boxes.append(parts)
        except ValueError as exc:
            err = MedicalError(
                code="INPUT_SCHEMA_ERROR",
                message=f"Invalid box at index {idx} {raw!r}: {exc}",
                fix="Use --box x1,y1,x2,y2 (e.g. --box 10,20,100,200)",
            )
            _emit_err(err, json_)
            raise typer.Exit(3)

    # Load image
    try:
        image = Image.open(input).convert("RGB")
    except Exception as exc:
        err = MedicalError(code="INPUT_LOAD_ERROR", message=str(exc), fix="Check image file.")
        _emit_err(err, json_)
        raise typer.Exit(3)

    # Run inference via VisionModel
    try:
        from visionservex import VisionModel

        # MedSAM v1 is research/education-only (legal_review). The medical CLI is
        # an explicit research pathway, so it acknowledges the restriction here.
        model = VisionModel("medsam", use_mode="research", acknowledge_license_restrictions=True)
        predict_kwargs: dict = {}
        if parsed_boxes:
            predict_kwargs["boxes"] = parsed_boxes
        result = model.predict(image, **predict_kwargs)
    except Exception as exc:
        msg = str(exc)
        if "checkpoint" in msg.lower() or "download" in msg.lower() or "not found" in msg.lower():
            err = MedicalError(
                code="CHECKPOINT_REQUIRED",
                message=f"MedSAM checkpoint not cached: {exc}",
                fix="visionservex model pull medsam",
            )
            _emit_err(err, json_)
            raise typer.Exit(3)
        err = MedicalError(
            code="MEDSAM_ENGINE_ERROR",
            message=str(exc)[:300],
            fix="Check transformers version and wanglab/medsam-vit-base cache.",
        )
        _emit_err(err, json_)
        raise typer.Exit(4)

    # Save masks
    import numpy as np

    masks_saved = []
    for i, seg in enumerate(result.segments if hasattr(result, "segments") else []):
        if seg.mask is not None:
            mask_path = out / f"mask_{i:03d}.png"
            mask_img = Image.fromarray((seg.mask * 255).astype(np.uint8))
            mask_img.save(mask_path)
            masks_saved.append(
                {
                    "mask_path": str(mask_path),
                    "iou_score": seg.score,
                    "box": ([seg.box.x1, seg.box.y1, seg.box.x2, seg.box.y2] if seg.box else None),
                }
            )

    payload = {
        "model_id": "medsam",
        "input": str(input),
        "boxes": boxes_raw or None,
        "out": str(out),
        "masks_saved": masks_saved,
        "n_masks": len(masks_saved),
        "device": getattr(result, "device", "unknown"),
        "status": "ok" if masks_saved else "no_masks_predicted",
        "disclaimer": DISCLAIMER,
    }
    meta_path = out / "medsam_metadata.json"
    meta_path.write_text(json_mod.dumps(payload, indent=2))

    if draw and masks_saved:
        try:
            import numpy as np

            base = Image.open(input).convert("RGB")
            overlay = np.array(base).copy()
            for m_info in masks_saved:
                mp = m_info.get("mask_path")
                if mp:
                    mask_arr = np.array(Image.open(mp).convert("L")) > 127
                    overlay[mask_arr] = overlay[mask_arr] * 0.5 + np.array([0, 200, 50]) * 0.5
            draw.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(overlay.astype(np.uint8)).save(draw)
        except Exception as exc:
            console.print(f"[yellow]DRAW_FAILED: {exc}[/yellow]")

    if json_:
        print(json_mod.dumps(payload, indent=2))
        return

    console.print("[green]MedSAM segmentation complete[/green]")
    console.print(f"  masks saved: {len(masks_saved)}")
    console.print(f"  metadata: {meta_path}")
    for m in masks_saved:
        console.print(f"  mask: {m['mask_path']} (IoU={m['iou_score']:.3f})")


@app.command("install-help")
def install_help_cmd(
    model_id: str = typer.Argument("", help="Medical model id (optional)."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Print install commands for medical models with license tier."""
    if model_id and model_id not in MEDICAL_MODELS:
        payload = {
            "code": "UNKNOWN_MODEL",
            "message": f"Unknown medical model {model_id!r}.",
            "available": list(MEDICAL_MODELS),
        }
        print(json.dumps(payload, indent=2)) if json_ else console.print(
            f"[red]Unknown model:[/red] {model_id}"
        )
        raise typer.Exit(2)

    targets = [model_id] if model_id else list(MEDICAL_MODELS)
    payload = {}
    for mid in targets:
        info = MEDICAL_MODELS[mid]
        payload[mid] = {
            "install": list(info.install),
            "license_tier": info.license_tier,
            "runtime_status": info.runtime_status,
            "checkpoint_status": info.checkpoint_status,
            "license_note": info.license_note,
            "structured_error_code": info.structured_error_code,
        }
    if json_:
        print(json.dumps(payload, indent=2))
        return
    for mid, entry in payload.items():
        console.print(f"\n[bold]{mid}[/bold]  [{entry['license_tier']}]")
        for cmd in entry["install"]:
            console.print(f"  [cyan]{cmd}[/cyan]")
        console.print(f"  license: {entry['license_note']}")


monai_app = typer.Typer(help="MONAI medical framework helpers.", no_args_is_help=True)


@monai_app.command("list-bundles")
def monai_list_bundles(
    fmt: str = typer.Option("text", "--format", help="Output format: text or json."),
    out: Path | None = typer.Option(None, "--out", help="Write JSON output to this path."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """List installed MONAI Model Zoo bundles (requires `pip install monai`)."""
    if not _module_present("monai"):
        payload: dict = {
            "status": "expected_blocker",
            "code": "MONAI_REQUIRED",
            "message": "monai is not installed.",
            "install_command": "pip install monai",
            "docs": "",
            "warnings": [],
            "errors": [],
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2))
        if json_ or fmt == "json":
            print(json.dumps(payload, indent=2))
        else:
            console.print(f"[yellow]{payload['code']}[/yellow] — {payload['install_command']}")
        # v2.19.0: expected_blocker exits 0 (matches sam-family validate / anomaly doctor).
        return
    try:
        from monai.bundle import get_all_bundles_list  # type: ignore

        bundles = get_all_bundles_list()
    except Exception as exc:  # pragma: no cover - real env
        payload = {
            "status": "failed",
            "code": "MONAI_API_ERROR",
            "message": str(exc)[:200],
            "install_command": "",
            "docs": "",
            "warnings": [],
            "errors": [str(exc)[:200]],
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2))
        if json_ or fmt == "json":
            print(json.dumps(payload, indent=2))
        else:
            console.print(f"[red]{payload['code']}[/red] — {payload['message']}")
        raise typer.Exit(4) from exc
    result_payload = {"status": "ok", "code": "", "bundles": bundles, "warnings": [], "errors": []}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result_payload, indent=2))
    if json_ or fmt == "json":
        print(json.dumps(result_payload, indent=2))
        return
    for b in bundles:
        console.print(f"  - {b}")


autoseg_app = typer.Typer(help="MONAI Auto3DSeg helpers.", no_args_is_help=True)


@autoseg_app.command("doctor")
def autoseg_doctor(json_: bool = typer.Option(False, "--json")) -> None:
    """Probe Auto3DSeg (MONAI) availability."""
    monai_ok = _module_present("monai")
    payload = {"monai_installed": monai_ok}
    if not monai_ok:
        payload.update(
            {
                "code": "AUTO3DSEG_REQUIRED",
                "fix": "pip install monai",
            }
        )
    if json_:
        print(json.dumps(payload, indent=2))
        return
    if monai_ok:
        console.print("[green]monai installed[/green]")
    else:
        console.print(f"[yellow]{payload['code']}[/yellow] — {payload['fix']}")


app.add_typer(monai_app, name="monai")
app.add_typer(autoseg_app, name="autoseg")

# Experimental MedSAM2 real runtime + medical training-truth sub-apps (lazy import
# keeps the medical CLI import-light; neither pulls torch/sam2 at import time).
from visionservex.cli.medical_train_commands import app as medical_train_app  # noqa: E402
from visionservex.cli.medsam2_commands import app as medsam2_app  # noqa: E402

app.add_typer(medsam2_app, name="medsam2")
app.add_typer(medical_train_app, name="train")


def _emit_err(err: MedicalError, json_: bool) -> None:
    if json_:
        print(json.dumps(err.to_dict(), indent=2))
    else:
        console.print(f"[red]{err.code}[/red]: {err.message}")
        console.print(f"  fix: {err.fix}")


__all__ = ["DISCLAIMER", "MEDICAL_MODELS", "MedicalError", "MedicalModelInfo", "app"]
