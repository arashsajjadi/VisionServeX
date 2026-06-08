# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""CLI commands for INSID3 in-context segmentation (BYOT — DINOv3 backbone)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="insid3",
    help=(
        "INSID3 training-free in-context segmentation (BYOT — DINOv3 backbone). "
        "Paper: arXiv 2603.28480 | visinf/INSID3 | Apache-2.0 code + DINOv3 backbone."
    ),
    no_args_is_help=True,
)


@app.command("status")
def status():
    """Show INSID3 policy rows and DINOv3 backbone access status."""
    from visionservex.licensing.policy import get_policy

    typer.echo("INSID3 policy rows (family: insid3)")
    typer.echo("=" * 60)
    for mid in ("insid3-small", "insid3-base", "insid3-large"):
        pol = get_policy(mid)
        if pol is None:
            typer.echo(f"  {mid}: NOT FOUND IN POLICY")
            continue
        typer.echo(f"  {mid}")
        typer.echo(f"    hf_repo       : {pol.hf_repo}")
        typer.echo(f"    policy        : {pol.final_policy}")
        typer.echo(f"    can_ship_weights: {pol.can_ship_weights}")
        typer.echo(f"    aliases       : {list(pol.aliases)}")
    typer.echo("")
    typer.echo("Run 'visionservex insid3 doctor' to check DINOv3 backbone access.")


@app.command("doctor")
def doctor(
    model_id: str = typer.Option("insid3-large", help="INSID3 model variant to check."),
):
    """Check HF token + DINOv3 backbone accessibility for INSID3."""
    from visionservex import hf_auth as _H
    from visionservex.licensing.policy import get_policy, resolve_model_id

    canonical = resolve_model_id(model_id)
    pol = get_policy(canonical)
    if pol is None or pol.family != "insid3":
        typer.echo(f"Unknown INSID3 model: {model_id}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Checking INSID3 access for: {canonical}")
    typer.echo(f"  HF repo: {pol.hf_repo}")

    token = _H.hf_get_token()
    if token:
        typer.echo("  HF token: FOUND (source: cache/env/local file)")
    else:
        typer.echo("  HF token: NOT FOUND — run 'huggingface-cli login'")

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.auth_check(pol.hf_repo, token=token)
        typer.echo(f"  Backbone access: ACCESS_GRANTED ({pol.hf_repo})")
    except Exception as exc:
        typer.echo(f"  Backbone access: BLOCKED — {exc}", err=True)
        raise typer.Exit(1)

    typer.echo("  Doctor: PASS — INSID3 is ready to run.")


@app.command("run")
def run(
    query_image: str = typer.Argument(..., help="Path to query image (to be segmented)."),
    reference_image: str = typer.Argument(..., help="Path to reference demonstration image."),
    reference_mask: str = typer.Argument(..., help="Path to reference mask (grayscale PNG)."),
    model_id: str = typer.Option("insid3-large", help="INSID3 variant: small/base/large."),
    device: str = typer.Option("cpu", help="Device: cpu or cuda."),
    n_clusters: int = typer.Option(6, help="Agglomerative clustering count."),
    out_dir: Optional[str] = typer.Option(
        None, help="Output directory for pred_mask.png, overlay.png, metadata.json."
    ),
):
    """Run INSID3 in-context segmentation on a query image given a reference pair."""
    import json

    from visionservex.insid3_runtime import insid3_segment

    result = insid3_segment(
        query_image,
        reference_image,
        reference_mask,
        model_id=model_id,
        device=device,
        n_clusters=n_clusters,
        out_dir=out_dir,
    )

    if result.get("status") != "ok":
        typer.echo(f"BLOCKED: {result.get('state')} — {result.get('reason')}", err=True)
        if result.get("next_command"):
            typer.echo(f"Next: {result['next_command']}", err=True)
        raise typer.Exit(1)

    typer.echo(json.dumps(result, indent=2, default=str))
    if result.get("mask_area_px", 0) == 0:
        typer.echo(
            "WARNING: mask_area_px=0. Check that reference_mask covers the target region.",
            err=True,
        )


@app.command("correspond")
def correspond(
    image_a: str = typer.Argument(..., help="First image path."),
    image_b: str = typer.Argument(..., help="Second image path."),
    model_id: str = typer.Option("insid3-large", help="INSID3 variant: small/base/large."),
    device: str = typer.Option("cpu", help="Device: cpu or cuda."),
    out_dir: Optional[str] = typer.Option(None, help="Directory for correspondence heatmap."),
):
    """Compute DINOv3 feature correspondences between two images (exploratory)."""
    import json

    from visionservex import hf_auth as _H
    from visionservex.licensing.policy import get_policy, resolve_model_id

    canonical = resolve_model_id(model_id)
    pol = get_policy(canonical)
    if pol is None or pol.family != "insid3":
        typer.echo(f"Unknown INSID3 model: {model_id}", err=True)
        raise typer.Exit(1)

    try:
        _H.hf_require_user_accepted_license(canonical)
    except _H.HFLicenseError as exc:
        typer.echo(f"BLOCKED: {exc}", err=True)
        raise typer.Exit(1)

    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        typer.echo("Missing deps: pip install 'visionservex[hf]' Pillow", err=True)
        raise typer.Exit(1)

    token = _H.hf_get_token()
    repo = pol.hf_repo
    proc = AutoImageProcessor.from_pretrained(repo, token=token)
    model = AutoModel.from_pretrained(repo, token=token).eval()

    def _feats(path: str):
        img = Image.open(path).convert("RGB")
        inputs = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
        return out.last_hidden_state[0, 1:, :].float().cpu()

    fa = _feats(image_a)
    fb = _feats(image_b)
    fa_n = fa / (fa.norm(dim=1, keepdim=True) + 1e-8)
    fb_n = fb / (fb.norm(dim=1, keepdim=True) + 1e-8)
    sim_matrix = fa_n @ fb_n.T  # (Na, Nb)
    max_sim = float(sim_matrix.max().item())
    mean_sim = float(sim_matrix.mean().item())

    result = {
        "status": "ok",
        "model_id": canonical,
        "hf_repo": repo,
        "image_a": image_a,
        "image_b": image_b,
        "patches_a": fa.shape[0],
        "patches_b": fb.shape[0],
        "max_cosine_sim": round(max_sim, 4),
        "mean_cosine_sim": round(mean_sim, 4),
        "attribution_required": "Built with DINOv3",
        "warning": pol.warning_text,
    }

    if out_dir:
        import json as _json

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        meta_path = out_path / "correspondence_metadata.json"
        meta_path.write_text(_json.dumps(result, indent=2))
        result["saved_paths"] = {"metadata": str(meta_path)}

    typer.echo(json.dumps(result, indent=2, default=str))
