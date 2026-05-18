# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Domain-zoo CLI: list domains, recommend pipelines per goal."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Domain-specialized model recommendations.")


@app.command("list", help="List all available domains.")
def list_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    from visionservex.model_zoo import DOMAIN_ZOO, list_domains

    if json_:
        payload = {d: list(DOMAIN_ZOO[d].keys()) for d in list_domains()}
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print("[bold]Available domains:[/bold]")
    for d in list_domains():
        goals = list(DOMAIN_ZOO[d].keys())
        console.print(f"  [cyan]{d}[/cyan]: {', '.join(goals)}")


@app.command("recommend", help="Recommend a pipeline for a domain + goal.")
def recommend_cmd(
    domain: str = typer.Option(..., "--domain"),
    goal: str | None = typer.Option(None, "--goal"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.model_zoo import recommend_for_domain

    recipes = recommend_for_domain(domain, goal)
    if not recipes:
        console.print(f"[yellow]No recipes for domain={domain!r} goal={goal!r}[/yellow]")
        from visionservex.model_zoo import list_domains

        console.print(f"  Available: {', '.join(list_domains())}")
        raise typer.Exit(1)

    if json_:
        typer.echo(json.dumps([r.to_dict() for r in recipes], indent=2))
        return

    for recipe in recipes:
        from rich.panel import Panel

        runnable = (
            "[green]✓ runnable today[/green]"
            if recipe.runnable_today
            else "[yellow]⚠ roadmap / expert-sidecar[/yellow]"
        )
        console.print(
            Panel.fit(
                f"[bold]{recipe.domain}[/bold] / {recipe.goal}\n"
                f"{recipe.description}\n\n"
                f"Status: {runnable}",
                border_style="cyan",
            )
        )
        console.print("\n[bold]Pipeline:[/bold]")
        for step in recipe.pipeline:
            console.print(f"  {step}")
        console.print("\n[bold]Recommended models:[/bold]")
        for m in recipe.recommended_models:
            console.print(f"  • {m}")
        if recipe.install_commands:
            console.print("\n[bold]Install:[/bold]")
            for c in recipe.install_commands:
                console.print(f"  [cyan]$[/cyan] {c}")
        if recipe.quick_commands:
            console.print("\n[bold]Quick commands:[/bold]")
            for c in recipe.quick_commands:
                console.print(f"  [cyan]$[/cyan] {c}")
        if recipe.expected_hardware:
            console.print(f"\n[bold]Hardware:[/bold] {recipe.expected_hardware}")
        if recipe.limitations:
            console.print("\n[bold]Limitations:[/bold]")
            for lim in recipe.limitations:
                console.print(f"  [yellow]⚠[/yellow] {lim}")
        if recipe.license_notes:
            console.print("\n[bold]License notes:[/bold]")
            for note in recipe.license_notes:
                console.print(f"  [dim]{note}[/dim]")
        console.print()


@app.command("yolo26-competitors", help="Show YOLO26-X competitor recommendations.")
def yolo26_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="yolo26-competitors", goal=None, json_=json_)


@app.command("sam-family", help="Show SAM-family recommendations.")
def sam_family_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="sam-family", goal=None, json_=json_)


@app.command("medical", help="Show medical-domain recommendations (extras).")
def medical_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="medical", goal=None, json_=json_)


@app.command("agriculture", help="Show agriculture-domain recommendations.")
def agriculture_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="agriculture", goal=None, json_=json_)


@app.command("aerial", help="Show aerial / OBB recommendations.")
def aerial_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="aerial", goal=None, json_=json_)


@app.command("industrial", help="Show industrial anomaly recommendations.")
def industrial_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="industrial", goal=None, json_=json_)


@app.command("surveillance", help="Show surveillance / video-search recommendations.")
def surveillance_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="surveillance", goal=None, json_=json_)


@app.command("feature-intelligence", help="Show DINOv2 / feature-backbone recommendations.")
def feature_intelligence_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="feature-intelligence", goal=None, json_=json_)


@app.command("promptable", help="Show open-vocabulary / promptable recommendations.")
def promptable_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="promptable", goal=None, json_=json_)


@app.command(
    "export",
    help="Export domain zoo to markdown.",
)
def export_cmd(
    out: Path = typer.Option(Path("docs/domain_zoo.md"), "--out"),
) -> None:
    from visionservex.model_zoo import DOMAIN_ZOO

    lines = ["# VisionServeX Domain Zoo", ""]
    lines.append(
        "Curated pipelines per vertical. Each recipe is honest: status, install, "
        "hardware, license, and known limitations are documented.\n"
    )
    for domain in sorted(DOMAIN_ZOO):
        lines.append(f"## {domain}\n")
        for goal_key, recipe in DOMAIN_ZOO[domain].items():
            lines.append(f"### {goal_key}")
            run = "✓ runnable today" if recipe.runnable_today else "⚠ roadmap / expert"
            lines.append(f"_{recipe.description}_  ")
            lines.append(f"**Status:** {run}\n")
            lines.append("**Pipeline:**")
            for step in recipe.pipeline:
                lines.append(f"1. {step}")
            lines.append("\n**Recommended models:**")
            for m in recipe.recommended_models:
                lines.append(f"- `{m}`")
            if recipe.install_commands:
                lines.append("\n**Install:**")
                for c in recipe.install_commands:
                    lines.append(f"```\n{c}\n```")
            if recipe.limitations:
                lines.append("\n**Limitations:**")
                for lim in recipe.limitations:
                    lines.append(f"- {lim}")
            lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Exported domain zoo to {out}[/green]")


# ---------------------------------------------------------------------------
# v2.18.0: domain-specific benchmark candidates
# ---------------------------------------------------------------------------

_DOMAIN_BENCHMARK_CANDIDATES: dict[str, dict] = {
    "medical": {
        "domain": "medical",
        "task": "medical_segmentation",
        "rows": [
            {
                "model_id": "medsam",
                "model_family": "MedSAM",
                "task": "promptable_medical_segmentation",
                "dataset_required": "2d_medical_image_with_box_prompts",
                "accepted_dataset_formats": ["png+jpeg+box_json"],
                "metrics_supported": ["mask_count", "latency", "visual_overlay"],
                "metrics_not_supported_without_gt": ["dice", "iou", "mask_ap"],
                "benchmark_status": "smoke_only",
                "recommended_command": (
                    "visionservex medical segment medsam IMAGE.png --box X1,Y1,X2,Y2 --out OUT_DIR"
                ),
                "expected_blocker_code": None,
            },
            {
                "model_id": "totalsegmentator",
                "model_family": "TotalSegmentator",
                "task": "3d_ct_segmentation",
                "dataset_required": "ct_volume_nifti",
                "accepted_dataset_formats": ["nii", "nii.gz"],
                "metrics_supported": ["dice_per_class", "iou_per_class"],
                "metrics_not_supported_without_gt": [],
                "benchmark_status": "validate_only",
                "recommended_command": "visionservex medical validate totalsegmentator",
                "expected_blocker_code": "NIFTI_REQUIRED",
            },
            {
                "model_id": "nnunet-v2",
                "model_family": "nnU-Net v2",
                "task": "trained_medical_segmentation",
                "dataset_required": "nnunet_dataset_format",
                "accepted_dataset_formats": ["nnunet_v2_layout"],
                "metrics_supported": [],
                "metrics_not_supported_without_gt": ["dice", "iou"],
                "benchmark_status": "expected_blocker",
                "recommended_command": "visionservex medical validate nnunet-v2",
                "expected_blocker_code": "DEPENDENCY_REQUIRED",
            },
        ],
        "excluded": [],
    },
    "agriculture": {
        "domain": "agriculture",
        "task": "weed_or_crop_detection",
        "rows": [
            {
                "model_id": "dfine-s-o365-coco",
                "model_family": "D-FINE",
                "task": "weed_or_crop_detection_via_finetune",
                "dataset_required": "yolo_crop_weed_labels",
                "accepted_dataset_formats": ["yolo", "coco-json"],
                "metrics_supported": ["ap50", "ap75", "map50_95"],
                "metrics_not_supported_without_gt": [],
                "benchmark_status": "metric_ready",
                "recommended_command": (
                    "visionservex benchmark-agriculture --dataset DATASET --models dfine-s-o365-coco"
                ),
                "expected_blocker_code": "LABELS_REQUIRED_FOR_METRICS",
            },
            {
                "model_id": "grounding-dino-swin-b",
                "model_family": "Grounding DINO",
                "task": "zero_shot_weed_detection",
                "dataset_required": "images_with_class_prompts",
                "accepted_dataset_formats": ["folder+prompts"],
                "metrics_supported": ["detection_count", "visual_overlay"],
                "metrics_not_supported_without_gt": ["ap50"],
                "benchmark_status": "demo_only",
                "recommended_command": (
                    "visionservex open-vocab grounding-dino-swin-b IMAGE --prompt 'weed,crop'"
                ),
                "expected_blocker_code": None,
            },
            {
                "model_id": "agriclip",
                "model_family": "AgriCLIP",
                "task": "agriculture_embedding",
                "dataset_required": "agri_image_dataset",
                "accepted_dataset_formats": ["folder"],
                "metrics_supported": [],
                "metrics_not_supported_without_gt": ["retrieval_top_k"],
                "benchmark_status": "expected_blocker",
                "recommended_command": "visionservex agriculture model-card agriclip",
                "expected_blocker_code": "SIDECAR_REQUIRED",
            },
        ],
        "excluded": [],
    },
    "aerial": {
        "domain": "aerial",
        "task": "aerial_detection_or_obb",
        "rows": [
            {
                "model_id": "rtmdet-r-l",
                "model_family": "RTMDet-R (rotated)",
                "task": "obb_detection",
                "dataset_required": "dota_v1_or_v2_obb",
                "accepted_dataset_formats": ["dota"],
                "metrics_supported": ["obb_ap50", "obb_map50_95"],
                "metrics_not_supported_without_gt": [],
                "benchmark_status": "expected_blocker",
                "recommended_command": "visionservex openmmlab validate rtmdet-r-l",
                "expected_blocker_code": "SIDECAR_REQUIRED",
            },
            {
                "model_id": "dfine-s-o365-coco",
                "model_family": "D-FINE",
                "task": "aerial_hbb_detection_via_finetune",
                "dataset_required": "visdrone_or_xview_yolo",
                "accepted_dataset_formats": ["yolo", "coco-json"],
                "metrics_supported": ["ap50", "ap75", "map50_95"],
                "metrics_not_supported_without_gt": ["obb_ap50"],
                "benchmark_status": "metric_ready",
                "recommended_command": (
                    "visionservex benchmark-aerial --dataset DATASET --dataset-type visdrone "
                    "--models dfine-s-o365-coco"
                ),
                "expected_blocker_code": "DOTA_OR_OBB_LABELS_REQUIRED",
            },
        ],
        "excluded": [],
    },
    "industrial": {
        "domain": "industrial",
        "task": "anomaly_detection",
        "rows": [
            {
                "model_id": "patchcore",
                "model_family": "anomalib PatchCore",
                "task": "image_level_anomaly",
                "dataset_required": "mvtec_or_simple_normal_defect",
                "accepted_dataset_formats": ["mvtec", "simple"],
                "metrics_supported": ["image_auroc", "pixel_auroc"],
                "metrics_not_supported_without_gt": [],
                "benchmark_status": "metric_ready",
                "recommended_command": (
                    "visionservex benchmark-anomaly --dataset simple:DATASET --model patchcore"
                ),
                "expected_blocker_code": None,
            },
            {
                "model_id": "padim",
                "model_family": "anomalib PaDiM",
                "task": "image_level_anomaly",
                "dataset_required": "mvtec_or_simple_normal_defect",
                "accepted_dataset_formats": ["mvtec", "simple"],
                "metrics_supported": ["image_auroc"],
                "metrics_not_supported_without_gt": [],
                "benchmark_status": "metric_ready",
                "recommended_command": (
                    "visionservex benchmark-anomaly --dataset simple:DATASET --model padim"
                ),
                "expected_blocker_code": None,
            },
        ],
        "excluded": [],
    },
    "surveillance": {
        "domain": "surveillance",
        "task": "video_search_pipeline",
        "rows": [
            {
                "model_id": "owlv2-base-patch16",
                "model_family": "OWLv2 + tracker + embedder",
                "task": "video_search_open_vocab_pipeline",
                "dataset_required": "video_or_frames_folder",
                "accepted_dataset_formats": ["mp4", "folder_of_frames"],
                "metrics_supported": ["retrieval_top_k", "timeline"],
                "metrics_not_supported_without_gt": ["track_map", "id_switch_rate"],
                "benchmark_status": "smoke_only",
                "recommended_command": (
                    "visionservex video-search index VIDEO --detector owlv2-base-patch16 --out INDEX"
                ),
                "expected_blocker_code": "GT_TRACKS_OR_QUERY_LABELS_REQUIRED",
            },
            {
                "model_id": "siglip2-base-patch16-224",
                "model_family": "SigLIP2 embedder",
                "task": "video_search_text_query",
                "dataset_required": "video_or_frames_folder_plus_text_query",
                "accepted_dataset_formats": ["mp4", "folder"],
                "metrics_supported": ["cosine_similarity_topk"],
                "metrics_not_supported_without_gt": ["map@k"],
                "benchmark_status": "smoke_only",
                "recommended_command": "visionservex video-search query INDEX --query 'TEXT'",
                "expected_blocker_code": None,
            },
        ],
        "excluded": [],
    },
    "segmentation": {
        "domain": "segmentation",
        "task": "promptable_or_dense_segmentation",
        "rows": [
            {
                "model_id": "sam-base",
                "model_family": "SAM",
                "task": "promptable_segmentation",
                "dataset_required": "image_with_box_or_point_prompts_and_gt_masks_for_metrics",
                "accepted_dataset_formats": ["png+jpeg+prompts+gt_masks_optional"],
                "metrics_supported": ["mask_count", "latency"],
                "metrics_not_supported_without_gt": ["mask_iou", "mask_ap"],
                "benchmark_status": "smoke_only",
                "recommended_command": "visionservex segment sam-base IMAGE --box X1,Y1,X2,Y2",
                "expected_blocker_code": "GT_MASKS_REQUIRED",
            },
            {
                "model_id": "sam2.1-base",
                "model_family": "SAM2.1",
                "task": "promptable_segmentation",
                "dataset_required": "image_or_video_with_prompts_and_gt_masks_optional",
                "accepted_dataset_formats": ["image", "video"],
                "metrics_supported": ["mask_count"],
                "metrics_not_supported_without_gt": ["mask_iou", "mask_ap"],
                "benchmark_status": "smoke_only",
                "recommended_command": "visionservex sam-family validate sam2.1-base",
                "expected_blocker_code": "GT_MASKS_REQUIRED",
            },
            {
                "model_id": "sam3.1",
                "model_family": "SAM3.1",
                "task": "gated_promptable_segmentation",
                "dataset_required": "n/a",
                "accepted_dataset_formats": [],
                "metrics_supported": [],
                "metrics_not_supported_without_gt": [],
                "benchmark_status": "expected_blocker",
                "recommended_command": "visionservex sam-family validate sam3.1",
                "expected_blocker_code": "GATED_HF_AUTH_REQUIRED",
            },
        ],
        "excluded": [],
    },
}

_DOMAIN_BENCHMARK_VALID = tuple(_DOMAIN_BENCHMARK_CANDIDATES.keys())


@app.command(
    "benchmark-candidates",
    help=(
        "v2.18.0: emit benchmark-eligibility candidates per domain "
        "(medical / agriculture / aerial / industrial / surveillance / segmentation)."
    ),
)
def domain_benchmark_candidates(
    domain: str = typer.Option(
        ..., "--domain", help=f"One of: {', '.join(_DOMAIN_BENCHMARK_VALID)}"
    ),
    out: Path = typer.Option(None, "--out", help="Write candidate JSON to this path."),
    fmt: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    """Single source of truth for which models can be benchmarked in a given domain.

    Detection candidates have their own command: `visionservex benchmark candidates`.
    """
    domain_lower = domain.lower()
    if domain_lower not in _DOMAIN_BENCHMARK_CANDIDATES:
        payload = {
            "status": "failed",
            "code": "UNKNOWN_DOMAIN",
            "domain": domain,
            "valid_domains": list(_DOMAIN_BENCHMARK_VALID),
            "message": f"Unknown domain {domain!r}. Valid: {list(_DOMAIN_BENCHMARK_VALID)}",
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2))
        if fmt == "json":
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(f"[red]UNKNOWN_DOMAIN[/red]: {payload['message']}")
        raise typer.Exit(2)

    payload = dict(_DOMAIN_BENCHMARK_CANDIDATES[domain_lower])
    payload["status"] = "ok"
    payload["code"] = "OK"
    payload["n_rows"] = len(payload["rows"])
    payload["n_excluded"] = len(payload["excluded"])

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(
        f"[bold]Domain candidates[/bold] (domain={domain_lower}): "
        f"{payload['n_rows']} eligible / {payload['n_excluded']} excluded."
    )
    for row in payload["rows"]:
        status_color = {
            "metric_ready": "green",
            "smoke_only": "yellow",
            "demo_only": "yellow",
            "validate_only": "yellow",
            "expected_blocker": "red",
        }.get(row["benchmark_status"], "white")
        console.print(
            f"  [{status_color}]{row['model_id']}[/{status_color}] "
            f"({row['benchmark_status']}) — needs {row['dataset_required']}"
        )


__all__ = ["app"]
