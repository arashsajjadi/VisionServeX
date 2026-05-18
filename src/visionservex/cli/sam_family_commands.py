# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""SAM-family CLI commands.

Covers: SAM v1, SAM 2, SAM 2.1, SAM 3, EfficientSAM, MobileSAM, FastSAM,
HQ-SAM, EdgeSAM.

For runnable variants (sam/sam2/sam2.1), uses VisionModel(MODEL_ID).predict().
For non-runnable variants, returns structured errors with recommended_action and
known_blockers from SOURCE_MANIFEST.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="SAM-family segment-anything model commands.", no_args_is_help=True)
console = Console()

_SAM_FAMILIES: dict[str, dict] = {
    "sam": {
        "license": "Apache-2.0",
        "install": "pip install 'visionservex[hf]'",
        "recommended_action": "add_now",
        "status": "runnable",
        "dep_modules": ["transformers"],
        "description": "SAM v1 — Segment Anything Model (ViT-B/L/H). Official HF checkpoints.",
    },
    "sam2": {
        "license": "Apache-2.0",
        "install": "pip install 'visionservex[hf]'",
        "recommended_action": "add_now",
        "status": "runnable",
        "dep_modules": ["transformers"],
        "description": "SAM 2 — video/image prompted segmentation (hiera backbone).",
    },
    "sam2.1": {
        "license": "Apache-2.0",
        "install": "pip install 'visionservex[hf]'",
        "recommended_action": "add_now",
        "status": "runnable",
        "dep_modules": ["transformers"],
        "description": "SAM 2.1 — improved SAM 2 with updated checkpoints.",
    },
    "sam3": {
        "license": "Apache-2.0",
        "install": "# gated — see https://huggingface.co/facebook/sam3",
        "recommended_action": "external_api",
        "status": "unavailable",
        "dep_modules": [],
        "description": "SAM 3 — gated release; not auto-pulled by VisionServeX.",
    },
    "efficientsam": {
        "license": "Apache-2.0",
        "install": "pip install efficientsam  # or: git clone https://github.com/yformer/EfficientSAM",
        "recommended_action": "expert_sidecar",
        "status": "optional_extra",
        "dep_modules": ["efficientsam"],
        "description": "EfficientSAM — distilled lightweight SAM (CVPR 2024).",
    },
    "mobilesam": {
        "license": "Apache-2.0",
        "install": "pip install mobile-sam",
        "recommended_action": "expert_sidecar",
        "status": "optional_extra",
        "dep_modules": ["mobile_sam"],
        "description": "MobileSAM — tiny ViT-based SAM for edge devices.",
    },
    "fastsam": {
        "license": "AGPL-3.0",
        "install": "# AGPL-3.0 — excluded from permissive core; see https://github.com/CASIA-IVA-Lab/FastSAM",
        "recommended_action": "do_not_add",
        "status": "unavailable",
        "dep_modules": [],
        "description": "FastSAM — AGPL-3.0 license; excluded from permissive Apache/MIT core.",
    },
    "hq-sam": {
        "license": "Apache-2.0",
        "install": "pip install segment-anything-hq",
        "recommended_action": "expert_sidecar",
        "status": "optional_extra",
        "dep_modules": ["segment_anything_hq"],
        "description": "HQ-SAM — high-quality mask prediction for complex structures.",
    },
    "edgesam": {
        "license": "Apache-2.0",
        "install": "git clone https://github.com/chongzhou96/EdgeSAM && pip install -e .",
        "recommended_action": "expert_sidecar",
        "status": "optional_extra",
        "dep_modules": ["edgesam"],
        "description": "EdgeSAM — real-time SAM for edge/mobile deployment.",
    },
}

_RUNNABLE_FAMILIES = {"sam", "sam2", "sam2.1"}


def _module_present(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def _sam_models_from_manifest() -> list:
    """Return all SAM-family entries from SOURCE_MANIFEST."""
    from visionservex.model_zoo import SOURCE_MANIFEST

    sam_families = set(_SAM_FAMILIES.keys())
    return [entry for entry in SOURCE_MANIFEST.values() if entry.family in sam_families]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("list")
def list_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    """List all SAM-family models from the manifest."""
    from visionservex.model_zoo import SOURCE_MANIFEST

    sam_families = set(_SAM_FAMILIES.keys())
    entries = [e for e in SOURCE_MANIFEST.values() if e.family in sam_families]

    if json_:
        print(json.dumps([e.to_dict() for e in entries], indent=2))
        return

    table = Table(title=f"SAM-family models ({len(entries)})", show_header=True)
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Family", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("License", no_wrap=True)
    table.add_column("Install")
    for e in entries:
        fam_info = _SAM_FAMILIES.get(e.family, {})
        status = fam_info.get("status", e.recommended_action)
        run_label = (
            "[green]runnable[/green]" if e.runnable_in_visionservex else f"[dim]{status}[/dim]"
        )
        table.add_row(
            e.model_id,
            e.family,
            run_label,
            e.license,
            e.install_command,
        )
    console.print(table)


@app.command("doctor")
def doctor_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    """Check which SAM variants have their dependencies installed."""
    results = []
    for family, info in _SAM_FAMILIES.items():
        deps = info.get("dep_modules", [])
        missing = [m for m in deps if not _module_present(m)]
        entry: dict = {
            "family": family,
            "status": info["status"],
            "deps": deps,
            "missing": missing,
            "installed": not missing,
            "install_command": info["install"],
            "license": info["license"],
        }
        if missing:
            entry["code"] = "SAM_EXTRA_REQUIRED"
            entry["fix"] = info["install"]
        results.append(entry)

    if json_:
        print(json.dumps(results, indent=2))
        return

    table = Table(title="SAM family dependency check", show_header=True)
    table.add_column("Family", style="cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Installed", no_wrap=True)
    table.add_column("Missing deps")
    for entry in results:
        installed_label = "[green]yes[/green]" if entry["installed"] else "[red]no[/red]"
        missing_label = ", ".join(entry["missing"]) if entry["missing"] else "-"
        table.add_row(
            entry["family"],
            entry["status"],
            installed_label,
            missing_label,
        )
    console.print(table)


@app.command("model-card")
def model_card_cmd(
    model_id: str = typer.Argument(..., help="SAM model ID (e.g. sam-vit-base)."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Print the SOURCE_MANIFEST entry for a SAM model."""
    from visionservex.model_zoo import get_model_source

    src = get_model_source(model_id)
    if src is None:
        err = {
            "code": "MODEL_NOT_FOUND",
            "message": f"Model {model_id!r} not found in SOURCE_MANIFEST.",
            "fix": "Run 'visionservex sam-family list' to see available models.",
        }
        if json_:
            print(json.dumps(err, indent=2))
        else:
            console.print(f"[red]MODEL_NOT_FOUND[/red]: {err['message']}")
            console.print(f"  fix: {err['fix']}")
        raise typer.Exit(1)

    # Verify it's a SAM family model
    if src.family not in _SAM_FAMILIES:
        err = {
            "code": "NOT_SAM_FAMILY",
            "message": f"Model {model_id!r} has family {src.family!r}, not a SAM variant.",
            "fix": "Run 'visionservex sam-family list' for SAM models.",
        }
        if json_:
            print(json.dumps(err, indent=2))
        else:
            console.print(f"[yellow]NOT_SAM_FAMILY[/yellow]: {err['message']}")
        raise typer.Exit(1)

    data = src.to_dict()
    if json_:
        print(json.dumps(data, indent=2))
        return

    console.print(f"[bold]{src.model_id}[/bold]  ({src.family}, {src.task})")
    if src.official_repo:
        console.print(f"  Repo:       {src.official_repo}")
    if src.hf_repo:
        console.print(f"  HF:         {src.hf_repo}")
    if src.paper_url:
        console.print(f"  Paper:      {src.paper_url}")
    console.print(f"  License:    {src.license} ({src.license_risk})")
    console.print(f"  Install:    {src.install_command}")
    console.print(f"  Runnable:   {src.runnable_in_visionservex}")
    console.print(f"  Action:     {src.recommended_action}")
    if src.known_blockers:
        console.print("  Blockers:")
        for b in src.known_blockers:
            console.print(f"    - {b}")
    if src.notes:
        console.print(f"  Notes:      {src.notes}")


@app.command("login-help")
def login_help_cmd(
    model_id: str = typer.Argument("sam3.1", help="SAM3/SAM3.1 model ID."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Print Hugging Face authentication steps for gated SAM3 / SAM3.1 weights.

    SAM3 / SAM3.1 weights are gated on Hugging Face. VisionServeX must NOT
    auto-pull them. This command prints the exact user-side commands.
    """
    payload = {
        "model_id": model_id,
        "code": "GATED_HF_AUTH_REQUIRED",
        "license": "Apache-2.0 (code), gated HF access (weights)",
        "official_repo": "https://github.com/facebookresearch/sam3",
        "paper": "https://arxiv.org/abs/2511.16719",
        "steps": [
            "1. Visit the HF model page and request access (e.g. facebook/sam3.1-hiera-base).",
            "2. pip install -U 'huggingface_hub[cli]'",
            "3. huggingface-cli login   # paste your HF token (read access).",
            "4. visionservex sam-family validate sam3.1",
            "5. Once approved, visionservex sam-family smoke-test sam3.1 IMAGE --box x1,y1,x2,y2",
        ],
        "warning": (
            "SAM3 / SAM3.1 are gated. VisionServeX will not bypass HF auth — "
            "do not attempt unauthenticated downloads."
        ),
    }
    if json_:
        print(json.dumps(payload, indent=2))
        return
    console.print(f"[bold]{model_id}[/bold] requires HF authentication.")
    for step in payload["steps"]:
        console.print(f"  {step}")
    console.print(f"\n[dim]Source: {payload['official_repo']}[/dim]")
    console.print(f"[yellow]{payload['warning']}[/yellow]")


@app.command("validate")
def validate_cmd(
    model_id: str = typer.Argument(...),
    fmt: str = typer.Option("text", "--format", help="Output format: text or json."),
    out: Path | None = typer.Option(None, "--out", help="Write structured JSON to this path."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Structured validation of a SAM-family model's dependencies / gated status."""
    from visionservex.model_zoo import get_model_source

    src = get_model_source(model_id)
    if src is None:
        # v2.16.0: SAM3 / SAM3.1 IDs that aren't in the manifest are still
        # the gated-auth blocker the user already understands. Returning
        # MODEL_NOT_FOUND with exit-2 caused the notebook to misclassify
        # `sam3.1` as a hard failure instead of an expected_blocker.
        lid = model_id.lower()
        if "sam3" in lid or "sam-3" in lid:
            payload = {
                "model_id": model_id,
                "family": "sam3",
                "status": "expected_blocker",
                "code": "GATED_HF_AUTH_REQUIRED",
                "message": (
                    f"{model_id!r} is a gated SAM3-family checkpoint. "
                    "VisionServeX does not auto-pull gated weights."
                ),
                "install_command": "",
                "docs": "visionservex sam-family login-help sam3.1",
                "fix": "visionservex sam-family login-help sam3.1",
                "known_blockers": [
                    "HuggingFace gated access required",
                    "VisionServeX will not bypass HF auth",
                ],
                "warnings": [],
                "errors": [],
            }
            if out:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(payload, indent=2))
            if json_ or fmt == "json":
                print(json.dumps(payload, indent=2))
            else:
                console.print(f"[yellow]{payload['code']}[/yellow]: {payload['message']}")
            # Exit 0 — expected blocker is not a failure for CI.
            return

        payload = {
            "model_id": model_id,
            "status": "expected_blocker",
            "code": "MODEL_NOT_FOUND",
            "message": f"Model {model_id!r} not in SOURCE_MANIFEST.",
            "install_command": "",
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
            console.print(f"[red]MODEL_NOT_FOUND[/red]: {payload['message']}")
        raise typer.Exit(2)

    payload = {
        "model_id": model_id,
        "family": src.family,
        "license": src.license,
        "license_risk": src.license_risk,
        "runnable": src.runnable_in_visionservex,
        "recommended_action": src.recommended_action,
        "known_blockers": list(src.known_blockers or []),
        "install_command": src.install_command,
        "official_repo": src.official_repo,
        "hf_repo": src.hf_repo,
        "warnings": [],
        "errors": [],
    }
    if src.family in {"sam3"} or "sam3" in model_id:
        payload["code"] = "GATED_HF_AUTH_REQUIRED"
        payload["status"] = "expected_blocker"
        payload["message"] = "SAM 3/3.1 requires HuggingFace gated access."
        payload["docs"] = "visionservex sam-family login-help sam3.1"
        payload["fix"] = "visionservex sam-family login-help sam3.1"
    elif not src.runnable_in_visionservex:
        payload["code"] = "MODEL_NOT_RUNNABLE"
        payload["status"] = "expected_blocker"
        payload["message"] = f"Model {model_id!r} is not runnable in VisionServeX core."
        payload["docs"] = src.official_repo or ""
    else:
        payload["code"] = "OK"
        payload["status"] = "ok"
        payload["message"] = f"Model {model_id!r} is available."
        payload["docs"] = src.official_repo or ""
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if json_ or fmt == "json":
        print(json.dumps(payload, indent=2))
        return
    console.print(f"[bold]{model_id}[/bold] — {payload['code']}")
    for key in ("license", "license_risk", "recommended_action"):
        console.print(f"  {key}: {payload[key]}")
    for b in payload["known_blockers"]:
        console.print(f"  blocker: {b}")


@app.command("smoke-test")
def smoke_test_cmd(
    model_id: str = typer.Argument(..., help="SAM model ID to test."),
    image: Path = typer.Argument(..., help="Input image path."),
    box: str = typer.Option(
        "",
        "--box",
        help="Box prompt 'x1,y1,x2,y2'.",
    ),
    out: Path | None = typer.Option(None, "--out", help="v2.19.0: save result JSON to this path."),
    draw: Path | None = typer.Option(
        None, "--draw", help="v2.19.0: save annotated image to this path."
    ),
    fmt: str = typer.Option(
        "text", "--format", help="Output format: text or json (notebook contract)."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Smoke-test a SAM model: run inference if runnable, else return structured error."""
    from visionservex.model_zoo import get_model_source

    json_mode = json_ or fmt == "json"

    def _emit_payload(payload: dict, exit_code: int = 0) -> None:
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2))
        if json_mode:
            print(json.dumps(payload, indent=2))
        if exit_code:
            raise typer.Exit(exit_code)

    src = get_model_source(model_id)
    if src is None:
        err = {
            "code": "MODEL_NOT_FOUND",
            "status": "expected_blocker",
            "message": f"Model {model_id!r} not found in SOURCE_MANIFEST.",
            "fix": "Run 'visionservex sam-family list' to see available models.",
        }
        if not json_mode:
            console.print(f"[red]MODEL_NOT_FOUND[/red]: {err['message']}")
        _emit_payload(err, exit_code=1)

    if not src.runnable_in_visionservex:
        fam_info = _SAM_FAMILIES.get(src.family, {})
        err = {
            "code": "MODEL_NOT_RUNNABLE",
            "status": "expected_blocker",
            "message": f"Model {model_id!r} is not runnable in this build.",
            "recommended_action": src.recommended_action,
            "known_blockers": src.known_blockers,
            "install": fam_info.get("install", src.install_command),
            "fix": (
                f"Install required deps: {fam_info.get('install', src.install_command)}"
                if fam_info.get("status") == "optional_extra"
                else "See known_blockers for details."
            ),
        }
        if not json_mode:
            console.print(f"[yellow]MODEL_NOT_RUNNABLE[/yellow]: {err['message']}")
            for b in src.known_blockers:
                console.print(f"  blocker: {b}")
            console.print(f"  action:  {src.recommended_action}")
        # v2.19.0: expected_blocker exits 0 (matches sam-family validate).
        _emit_payload(err, exit_code=0)
        return

    if not image.exists():
        err = {
            "code": "INPUT_NOT_FOUND",
            "status": "failed",
            "message": f"Image not found: {image}",
            "fix": f"Check path: {image}",
        }
        if not json_mode:
            console.print(f"[red]INPUT_NOT_FOUND[/red]: {err['message']}")
        _emit_payload(err, exit_code=3)
        return

    # Parse box
    boxes = None
    if box:
        try:
            parts = [float(x.strip()) for x in box.split(",")]
            if len(parts) != 4:
                raise ValueError(f"Expected x1,y1,x2,y2 got {len(parts)} values")
            boxes = [parts]
        except ValueError as exc:
            err = {
                "code": "INPUT_SCHEMA_ERROR",
                "status": "failed",
                "message": f"Invalid box format {box!r}: {exc}",
                "fix": "Use --box x1,y1,x2,y2",
            }
            if not json_mode:
                console.print(f"[red]INPUT_SCHEMA_ERROR[/red]: {err['message']}")
            _emit_payload(err, exit_code=3)
            return

    # Run inference
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
            "box": box or None,
            "status": "ok",
            "code": "OK",
            "n_segments": n_segments,
            "family": src.family,
        }
        # v2.19.0: optional --draw overlay
        if draw is not None and hasattr(result, "save_image"):
            try:
                draw.parent.mkdir(parents=True, exist_ok=True)
                result.save_image(draw)
                payload["draw_path"] = str(draw)
            except Exception as draw_exc:
                payload.setdefault("warnings", []).append(f"draw_failed: {draw_exc!s:.120}")
        if not json_mode:
            console.print(f"[green]smoke-test passed[/green]: {model_id}")
            console.print(f"  segments: {n_segments}")
        _emit_payload(payload)
    except Exception as exc:
        err = {
            "code": "SMOKE_TEST_ERROR",
            "status": "failed",
            "message": str(exc)[:300],
            "fix": "Check transformers version and HF cache.",
        }
        if not json_mode:
            console.print(f"[red]SMOKE_TEST_ERROR[/red]: {err['message']}")
        _emit_payload(err, exit_code=4)


__all__ = ["app"]
