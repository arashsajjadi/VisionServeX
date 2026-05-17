# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Model zoo CLI: sources, verify-links, export."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="Source-grounded model manifest commands.")


@app.command("sources", help="List all model sources with URLs and license.")
def sources_cmd(
    runnable_only: bool = typer.Option(False, "--runnable-only"),
    domain: str | None = typer.Option(None, "--domain"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.model_zoo import SOURCE_MANIFEST

    entries = list(SOURCE_MANIFEST.values())
    if runnable_only:
        entries = [e for e in entries if e.runnable_in_visionservex]
    if domain:
        entries = [e for e in entries if e.domain.lower() == domain.lower()]

    if json_:
        typer.echo(json.dumps([e.to_dict() for e in entries], indent=2))
        return

    table = Table(title=f"Model sources ({len(entries)})")
    for col in ("Model ID", "Family", "Task", "Runnable", "Access", "Action", "License"):
        table.add_column(col)
    for e in entries:
        run = "[green]yes[/green]" if e.runnable_in_visionservex else "[grey50]no[/grey50]"
        access = e.access_status
        access_color = {"open": "green", "api_token": "yellow", "gated": "red"}.get(access, "white")
        action_color = {
            "add_now": "green",
            "expert_sidecar": "yellow",
            "external_api": "magenta",
            "audit_only": "grey50",
            "do_not_add": "red",
            "non_core_license_optional": "yellow",
        }.get(e.recommended_action, "white")
        table.add_row(
            e.model_id,
            e.family,
            e.task,
            run,
            f"[{access_color}]{access}[/{access_color}]",
            f"[{action_color}]{e.recommended_action}[/{action_color}]",
            f"{e.license}{' ⚠' if e.license_risk not in ('none', '') else ''}",
        )
    console.print(table)


@app.command("verify-links", help="Static verification of manifest (no network calls).")
def verify_links_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    from visionservex.model_zoo import verify_manifest

    report = verify_manifest()
    if json_:
        typer.echo(json.dumps(report, indent=2))
        return
    counts = report["counts"]
    console.print("[bold]Manifest verification[/bold]")
    console.print(f"  Total entries:       {counts['total']}")
    console.print(f"  Runnable:            [green]{counts['runnable']}[/green]")
    console.print(f"  Expert sidecar:      [yellow]{counts.get('expert_sidecar', 0)}[/yellow]")
    console.print(f"  External API:        [magenta]{counts.get('external_api', 0)}[/magenta]")
    console.print(f"  Audit only:          [grey50]{counts.get('audit_only', 0)}[/grey50]")
    console.print(
        f"  Non-core license:    [yellow]{counts.get('non_core_license_optional', 0)}[/yellow]"
    )
    console.print(f"  Do not add:          [red]{counts.get('do_not_add', 0)}[/red]")
    if report["issues"]:
        console.print("\n[yellow]Issues:[/yellow]")
        for issue in report["issues"]:
            console.print(f"  {issue['model_id']}: {issue['issue']}")
    else:
        console.print("\n[green]No structural issues found.[/green]")


@app.command("export", help="Export manifest to JSON or markdown.")
def export_cmd(
    format_: str = typer.Option("json", "--format", help="json | markdown"),
    out: Path = typer.Option(Path("docs/model_zoo_manifest.json"), "--out"),
) -> None:
    from visionservex.model_zoo import SOURCE_MANIFEST

    out.parent.mkdir(parents=True, exist_ok=True)
    if format_ == "json":
        payload = {mid: src.to_dict() for mid, src in SOURCE_MANIFEST.items()}
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif format_ == "markdown":
        lines = ["# VisionServeX Model Zoo Manifest", ""]
        lines.append("| Model ID | Family | Task | Runnable | Access | License | Action |")
        lines.append("|----------|--------|------|----------|--------|---------|--------|")
        for _mid, src in sorted(SOURCE_MANIFEST.items()):
            run = "✓" if src.runnable_in_visionservex else "—"
            lines.append(
                f"| `{src.model_id}` | {src.family} | {src.task} | {run} | "
                f"{src.access_status} | {src.license} | {src.recommended_action} |"
            )
        lines.append("")
        lines.append("## Sources")
        for _mid, src in sorted(SOURCE_MANIFEST.items()):
            lines.append(f"### `{src.model_id}`")
            if src.official_repo:
                lines.append(f"- Official: <{src.official_repo}>")
            if src.hf_repo:
                lines.append(f"- HF: `{src.hf_repo}`")
            if src.paper_url:
                lines.append(f"- Paper: <{src.paper_url}>")
            if src.known_blockers:
                lines.append("- Blockers:")
                for b in src.known_blockers:
                    lines.append(f"  - {b}")
            lines.append("")
        out.write_text("\n".join(lines), encoding="utf-8")
    else:
        console.print(f"[red]unknown format: {format_}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Exported manifest to {out}[/green]")


@app.command("show", help="Show source detail for one model.")
def show_cmd(
    model_id: str,
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.model_zoo import get_model_source

    src = get_model_source(model_id)
    if src is None:
        console.print(f"[red]not in manifest:[/red] {model_id}")
        raise typer.Exit(1)
    if json_:
        typer.echo(json.dumps(src.to_dict(), indent=2))
        return
    console.print(f"[bold]{src.model_id}[/bold]  ({src.family}, {src.task})")
    if src.official_repo:
        console.print(f"  Repo:        {src.official_repo}")
    if src.official_docs:
        console.print(f"  Docs:        {src.official_docs}")
    if src.hf_repo:
        console.print(f"  HF:          {src.hf_repo}")
    if src.paper_url:
        console.print(f"  Paper:       {src.paper_url}")
    console.print(f"  License:     {src.license} ({src.license_risk})")
    console.print(f"  Install:     {src.install_command}")
    console.print(f"  Runnable:    {src.runnable_in_visionservex}")
    console.print(f"  Action:      {src.recommended_action}")
    if src.known_blockers:
        console.print("  Blockers:")
        for b in src.known_blockers:
            console.print(f"    - {b}")
    if src.notes:
        console.print(f"  Notes:       {src.notes}")


@app.command("gap-report")
def gap_report(
    format_: str = typer.Option("markdown", "--format"),
    out: Path = typer.Option(None, "--out"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Generate a gap report from SOURCE_MANIFEST grouped by recommended_action."""
    from visionservex.model_zoo import SOURCE_MANIFEST

    groups: dict[str, list] = {
        "runnable": [],
        "optional_extra": [],
        "expert_sidecar": [],
        "external_api": [],
        "do_not_add": [],
        "audit_only": [],
        "unavailable": [],
        "non_core_license_optional": [],
    }
    for entry in SOURCE_MANIFEST.values():
        action = entry.recommended_action
        if entry.runnable_in_visionservex and action == "add_now":
            groups["runnable"].append(entry)
        elif action == "expert_sidecar":
            groups["expert_sidecar"].append(entry)
        elif action == "external_api":
            groups["external_api"].append(entry)
        elif action == "do_not_add":
            groups["do_not_add"].append(entry)
        elif action == "non_core_license_optional":
            groups["non_core_license_optional"].append(entry)
        elif action == "audit_only":
            if entry.known_blockers:
                groups["unavailable"].append(entry)
            else:
                groups["audit_only"].append(entry)
        else:
            groups["audit_only"].append(entry)

    payload = {grp: [e.to_dict() for e in entries] for grp, entries in groups.items()}
    counts = {grp: len(entries) for grp, entries in groups.items()}
    payload["_counts"] = counts  # type: ignore[assignment]

    if json_:
        print(json.dumps(payload, indent=2))
        return

    if format_ == "json":
        text = json.dumps(payload, indent=2)
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            console.print(f"[green]Gap report written to {out}[/green]")
        else:
            print(text)
        return

    lines = ["# VisionServeX Model Zoo Gap Report", ""]
    section_labels = {
        "runnable": "Runnable (wired, usable now)",
        "optional_extra": "Optional extras (require extra install)",
        "expert_sidecar": "Expert sidecars (OpenMMLab, Detectron2, etc.)",
        "external_api": "External / gated APIs",
        "non_core_license_optional": "Non-core license (optional)",
        "do_not_add": "Excluded (do_not_add with reason)",
        "audit_only": "Audit only (no blockers yet)",
        "unavailable": "Unresolved blockers",
    }
    for grp, label in section_labels.items():
        entries = groups[grp]
        lines.append(f"## {label} ({len(entries)})")
        lines.append("")
        if entries:
            lines.append("| Model ID | Family | Task | License | Blockers |")
            lines.append("|----------|--------|------|---------|---------|")
            for e in entries:
                blockers = "; ".join(e.known_blockers[:2]) if e.known_blockers else "-"
                lines.append(
                    f"| `{e.model_id}` | {e.family} | {e.task} | {e.license} | {blockers} |"
                )
        else:
            lines.append("_None_")
        lines.append("")

    text = "\n".join(lines)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        console.print(f"[green]Gap report written to {out}[/green]")
    else:
        console.print(text)


@app.command("matrix")
def matrix(
    format_: str = typer.Option("markdown", "--format"),
    out: Path = typer.Option(None, "--out"),
    json_: bool = typer.Option(False, "--json"),
    family: str = typer.Option("", "--family", help="Filter by family"),
    domain: str = typer.Option("", "--domain", help="Filter by domain"),
) -> None:
    """Generate a full model matrix from SOURCE_MANIFEST."""
    from visionservex.model_zoo import SOURCE_MANIFEST

    entries = list(SOURCE_MANIFEST.values())
    if family:
        entries = [e for e in entries if e.family.lower() == family.lower()]
    if domain:
        entries = [e for e in entries if e.domain.lower() == domain.lower()]

    rows = []
    for e in entries:
        rows.append(
            {
                "model_id": e.model_id,
                "family": e.family,
                "task": e.task,
                "status": "runnable" if e.runnable_in_visionservex else e.recommended_action,
                "license": e.license,
                "install": e.install_command,
                "source_url": e.official_repo or e.hf_repo or "",
                "blockers": e.known_blockers,
            }
        )

    if json_:
        print(json.dumps(rows, indent=2))
        return

    if format_ == "json":
        text = json.dumps(rows, indent=2)
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            console.print(f"[green]Matrix written to {out}[/green]")
        else:
            print(text)
        return

    lines = ["# VisionServeX Model Matrix", ""]
    lines.append("| Model ID | Family | Task | Status | License | Install | Source | Blockers |")
    lines.append("|----------|--------|------|--------|---------|---------|--------|---------|")
    for row in rows:
        blockers = "; ".join(row["blockers"][:1]) if row["blockers"] else "-"
        lines.append(
            f"| `{row['model_id']}` | {row['family']} | {row['task']} | "
            f"{row['status']} | {row['license']} | `{row['install']}` | "
            f"{row['source_url'] or '-'} | {blockers} |"
        )
    lines.append("")

    text = "\n".join(lines)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        console.print(f"[green]Matrix written to {out}[/green]")
    else:
        console.print(text)


# v2.9 certified-blocker registry. Each entry encodes everything a release
# auditor needs to certify that an external blocker is real and stable:
# variants list, official_repo + source_files_checked, license + risk,
# the exact missing piece, a future_unblock_condition, and a
# blocker_certainty in [0, 100].
CERTIFIED_BLOCKERS: dict[str, dict] = {
    "dfine-native": {
        "family": "dfine",
        "variants": [
            "dfine-n",
            "dfine-s",
            "dfine-m",
            "dfine-l",
            "dfine-x",
        ],
        "official_repo": "https://github.com/Peterande/D-FINE",
        "paper": "https://arxiv.org/abs/2410.13861",
        "license": "Apache-2.0",
        "license_risk": "none",
        "install_route": (
            "git clone https://github.com/Peterande/D-FINE && pip install -r "
            "requirements.txt (no pip package on PyPI for the native loader)."
        ),
        "checkpoint_status": "official_release_assets",
        "loader_status": "native_only",
        "config_status": "native_only",
        "exact_missing_piece": (
            "No standardized pip package for the native loader. The HF "
            "Transformers route covers the runnable D-FINE variants; the "
            "native loader is sidecar-only."
        ),
        "tested_commands": [
            "git clone https://github.com/Peterande/D-FINE",
            "pip install -r D-FINE/requirements.txt",
            "python D-FINE/tools/inference.py -c configs/dfine/dfine_x_coco.yml",
        ],
        "source_files_checked": [
            "https://github.com/Peterande/D-FINE/blob/main/README.md",
            "https://github.com/Peterande/D-FINE/tree/main/configs",
        ],
        "date_checked": "2026-05-16",
        "future_unblock_condition": (
            "Upstream ships a pip-installable native runtime, or a HF "
            "Transformers Auto* class lands for D-FINE custom heads."
        ),
        "recommended_route": "hf_or_sidecar",
        "status": "optional_sidecar",
        "blocker_certainty": 96,
    },
    "rtdetrv4": {
        "family": "rtdetrv4",
        "variants": ["rtdetrv4-s", "rtdetrv4-m", "rtdetrv4-l", "rtdetrv4-x"],
        "official_repo": "https://github.com/RT-DETRs/RT-DETRv4",
        "paper": "https://arxiv.org/abs/2510.25257",
        "license": "Apache-2.0",
        "license_risk": "none",
        "install_route": (
            "git clone https://github.com/RT-DETRs/RT-DETRv4 && pip install "
            "-r requirements.txt (no clean HF/native pip loader)."
        ),
        "checkpoint_status": "release_assets",
        "loader_status": "native_only",
        "config_status": "native_only",
        "exact_missing_piece": (
            "RT-DETRv4 requires repo-internal config/checkpoint mapping and "
            "shows known TensorRT/sm_120 issues on the RTX 5080 wheel."
        ),
        "tested_commands": [
            "git clone https://github.com/RT-DETRs/RT-DETRv4",
            "pip install -r RT-DETRv4/requirements.txt",
            "python tools/inference/torch_inf.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml -r model.pth",
        ],
        "source_files_checked": [
            "https://github.com/RT-DETRs/RT-DETRv4/blob/main/README.md",
            "https://github.com/RT-DETRs/RT-DETRv4/tree/main/configs",
        ],
        "date_checked": "2026-05-16",
        "future_unblock_condition": (
            "Upstream publishes a pip package or a torch.hub entry; sm_120 "
            "regressions are resolved with a torch 2.6+ wheel."
        ),
        "recommended_route": "expert_sidecar",
        "status": "optional_sidecar",
        "blocker_certainty": 95,
    },
    "deimv2": {
        "family": "deimv2",
        "variants": ["deimv2-s", "deimv2-m", "deimv2-l", "deimv2-x"],
        "official_repo": "https://github.com/Intellindust-AI-Lab/DEIMv2",
        "paper": "https://huggingface.co/papers/2509.20787",
        "license": "Apache-2.0",
        "license_risk": "none",
        "install_route": (
            "git clone https://github.com/Intellindust-AI-Lab/DEIMv2 && "
            "follow the in-repo INSTALL guide."
        ),
        "checkpoint_status": "hf_hub_candidate",
        "loader_status": "native_only",
        "config_status": "native_only",
        "exact_missing_piece": (
            "DEIMv2 has no Transformers loader; native repo loader is the "
            "only path. HF hub IDs exist but the in-repo loader is the "
            "supported route."
        ),
        "tested_commands": [
            "git clone https://github.com/Intellindust-AI-Lab/DEIMv2",
            "pip install -r DEIMv2/requirements.txt",
        ],
        "source_files_checked": [
            "https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/README.md",
            "https://huggingface.co/Intellindust/DEIMv2_DINOv3_S_COCO",
        ],
        "date_checked": "2026-05-16",
        "future_unblock_condition": (
            "HF Transformers ships a DEIMv2 AutoModel, or the upstream repo "
            "publishes a `pip install deimv2` package."
        ),
        "recommended_route": "expert_sidecar",
        "status": "optional_sidecar",
        "blocker_certainty": 95,
    },
    "maskdino": {
        "family": "maskdino",
        "variants": list(
            __import__(
                "visionservex.cli.maskdino_commands", fromlist=["_MASKDINO_MODELS"]
            )._MASKDINO_MODELS
        ),
        "official_repo": "https://github.com/IDEA-Research/MaskDINO",
        "paper": "https://arxiv.org/abs/2206.02777",
        "license": "Apache-2.0",
        "license_risk": "none",
        "install_route": "bash scripts/run_maskdino_smoke.sh",
        "checkpoint_status": "official_release_assets",
        "loader_status": "detectron2_sidecar",
        "config_status": "config_in_repo",
        "exact_missing_piece": (
            "Detectron2 custom CUDA ops require the matching CUDA toolkit; "
            "VisionServeX core stays permissive-only."
        ),
        "tested_commands": [
            "bash scripts/run_maskdino_smoke.sh examples/images/street.jpg",
            "visionservex maskdino validate maskdino-swinl-coco --json",
        ],
        "source_files_checked": [
            "https://github.com/IDEA-Research/MaskDINO/blob/main/README.md",
            "https://github.com/IDEA-Research/detrex-storage/releases/tag/maskdino-v0.1.0",
        ],
        "date_checked": "2026-05-16",
        "future_unblock_condition": (
            "Detectron2 ships as a binary wheel for the current CUDA, OR "
            "MaskDINO is re-released against the mmcv 2.x / pure-PyTorch stack."
        ),
        "recommended_route": "expert_sidecar",
        "status": "optional_sidecar",
        "blocker_certainty": 97,
    },
    "co-dino": {
        "family": "co-dino",
        "variants": ["co-dino-inst-vit-l-coco", "co-dino-inst-vit-l-lvis"],
        "official_repo": "https://github.com/Sense-X/Co-DETR",
        "paper": "https://arxiv.org/abs/2211.12860",
        "license": "Apache-2.0",
        "license_risk": "none",
        "install_route": (
            "OpenMMLab sidecar (see scripts/run_openmmlab_rtmpose_smoke.sh "
            "and adapt the config for Co-DETR/Co-DINO)."
        ),
        "checkpoint_status": "official_upstream",
        "loader_status": "openmmlab_sidecar",
        "config_status": "config_in_repo",
        "exact_missing_piece": (
            "Co-DETR / Co-DINO configs target mmcv 2.x but require mmdet's "
            "registry to load custom heads. Same env as RTMPose/RTMDet."
        ),
        "tested_commands": [
            "visionservex openmmlab smoke-test co-dino-inst-vit-l-coco --device cpu",
        ],
        "source_files_checked": [
            "https://github.com/Sense-X/Co-DETR/blob/main/README.md",
        ],
        "date_checked": "2026-05-16",
        "future_unblock_condition": (
            "Add a per-model Co-DETR config alias to `_PULL_METADATA` in "
            "cli/openmmlab_commands.py with the official checkpoint URL."
        ),
        "recommended_route": "openmmlab_sidecar",
        "status": "optional_sidecar",
        "blocker_certainty": 92,
    },
    "dfine-seg": {
        "family": "dfine-seg",
        "variants": [],
        "official_repo": "https://github.com/Peterande/D-FINE",
        "paper": "https://arxiv.org/abs/2410.13861",
        "license": "Apache-2.0",
        "license_risk": "none",
        "install_route": "unavailable",
        "checkpoint_status": "not_released",
        "loader_status": "not_released",
        "config_status": "not_released",
        "exact_missing_piece": (
            "Upstream D-FINE has no released segmentation head. Anything "
            "claiming D-FINE-Seg is community / unofficial."
        ),
        "tested_commands": [],
        "source_files_checked": [
            "https://github.com/Peterande/D-FINE/blob/main/README.md",
        ],
        "date_checked": "2026-05-16",
        "future_unblock_condition": (
            "D-FINE-Seg upstream release with checkpoint and config files."
        ),
        "recommended_route": "unavailable_with_reason",
        "status": "unavailable_with_reason",
        "blocker_certainty": 97,
    },
    "di-maskdino": {
        "family": "di-maskdino",
        "variants": [],
        "official_repo": "https://arxiv.org/abs/2406.04302",
        "paper": "https://arxiv.org/abs/2406.04302",
        "license": "unknown",
        "license_risk": "unknown",
        "install_route": "unavailable",
        "checkpoint_status": "not_released",
        "loader_status": "not_released",
        "config_status": "not_released",
        "exact_missing_piece": ("DI-MaskDINO has no public code/checkpoint release at audit time."),
        "tested_commands": [],
        "source_files_checked": [
            "https://arxiv.org/abs/2406.04302",
        ],
        "date_checked": "2026-05-16",
        "future_unblock_condition": "Public repo + checkpoint release.",
        "recommended_route": "unavailable_with_reason",
        "status": "unavailable_with_reason",
        "blocker_certainty": 96,
    },
    "rfdetr-plus": {
        "family": "rfdetr",
        "variants": ["rfdetr-plus", "rfdetr-xl", "rfdetr-2xl"],
        "official_repo": "https://github.com/roboflow/rf-detr",
        "paper": "https://blog.roboflow.com/rf-detr/",
        "license": "PML 1.0",
        "license_risk": "restricted",
        "install_route": "pip install 'rfdetr[plus]'",
        "checkpoint_status": "package_managed",
        "loader_status": "vendor_loader",
        "config_status": "vendor_managed",
        "exact_missing_piece": (
            "RF-DETR Plus / XL / 2XL ship under Roboflow PML 1.0; not permissive-core safe."
        ),
        "tested_commands": [],
        "source_files_checked": [
            "https://github.com/roboflow/rf-detr/blob/main/LICENSE",
        ],
        "date_checked": "2026-05-16",
        "future_unblock_condition": ("Roboflow re-licenses Plus/XL/2XL under Apache-2.0 or MIT."),
        "recommended_route": "non_core_license_optional",
        "status": "non_core_license_optional",
        "blocker_certainty": 99,
    },
    "rfdetr-seg-large": {
        "family": "rfdetr",
        "variants": ["rfdetr-seg-large"],
        "official_repo": "https://github.com/roboflow/rf-detr",
        "paper": "https://blog.roboflow.com/rf-detr/",
        "license": "PML 1.0 (Plus tier)",
        "license_risk": "restricted",
        "install_route": "pip install 'rfdetr[plus]'",
        "checkpoint_status": "package_managed",
        "loader_status": "vendor_loader",
        "config_status": "vendor_managed",
        "exact_missing_piece": (
            "rfdetr-seg-large is part of the PML-1.0 Plus tier; not permissive-core safe."
        ),
        "tested_commands": [],
        "source_files_checked": [
            "https://github.com/roboflow/rf-detr/blob/main/LICENSE",
        ],
        "date_checked": "2026-05-16",
        "future_unblock_condition": (
            "Roboflow re-licenses Plus/XL/2XL or releases an Apache-2.0 rfdetr-seg-large variant."
        ),
        "recommended_route": "non_core_license_optional",
        "status": "non_core_license_optional",
        "blocker_certainty": 99,
    },
}


@app.command("blockers")
def blockers_cmd(
    family: str = typer.Option("", "--family"),
    all_: bool = typer.Option(
        False, "--all", help="Emit all certified blockers (no family filter)."
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Emit the certified blocker record for the family (when available).",
    ),
    out: Path = typer.Option(None, "--out", help="Write the JSON blocker report to this path."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Show known blockers for a model family.

    With ``--refresh`` and a known ``--family`` name, emit the full
    certified-blocker payload (license, source files checked, exact
    missing piece, future unblock condition, blocker certainty).
    """
    from visionservex.model_zoo import SOURCE_MANIFEST

    # --all: emit every certified blocker as a JSON array.
    if all_:
        payload_all = [{"family": fam, **cert} for fam, cert in CERTIFIED_BLOCKERS.items()]
        text = json.dumps(payload_all, indent=2)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text)
        if json_ or out is not None:
            print(text)
        else:
            for cert in payload_all:
                console.print(
                    f"[bold]{cert['family']}[/bold]  "
                    f"status={cert.get('status')}  "
                    f"certainty={cert.get('blocker_certainty')}%"
                )
        return

    if refresh and family:
        cert = CERTIFIED_BLOCKERS.get(family.lower())
        if cert is None:
            payload = {
                "code": "FAMILY_NOT_CERTIFIED",
                "family": family,
                "available_certifications": sorted(CERTIFIED_BLOCKERS),
            }
        else:
            payload = {"family": family, **cert}
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2))
            payload["out"] = str(out)
        if json_ or out is not None:
            print(json.dumps(payload, indent=2))
            return
        console.print(
            f"[bold]{family}[/bold]  status={cert.get('status') if cert else 'unknown'}"
            f"  certainty={cert.get('blocker_certainty') if cert else 'N/A'}%"
        )
        if cert:
            console.print(f"  missing: {cert['exact_missing_piece']}")
            console.print(f"  unblock: {cert['future_unblock_condition']}")
        return

    entries = list(SOURCE_MANIFEST.values())
    if family:
        entries = [e for e in entries if e.family.lower() == family.lower()]

    blocked = [e for e in entries if e.known_blockers]
    rows = []
    for e in blocked:
        rows.append(
            {
                "model_id": e.model_id,
                "family": e.family,
                "recommended_action": e.recommended_action,
                "known_blockers": e.known_blockers,
                "install": e.install_command,
            }
        )

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2))

    if json_:
        print(json.dumps(rows, indent=2))
        return

    if not rows:
        label = f" for family {family!r}" if family else ""
        console.print(f"[green]No blockers found{label}.[/green]")
        return

    table = Table(title=f"Known blockers ({len(rows)})", show_header=True)
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Family", no_wrap=True)
    table.add_column("Action", no_wrap=True)
    table.add_column("Blockers")
    for row in rows:
        blockers_text = "\n".join(f"- {b}" for b in row["known_blockers"])
        table.add_row(
            row["model_id"],
            row["family"],
            row["recommended_action"],
            blockers_text,
        )
    console.print(table)


__all__ = ["CERTIFIED_BLOCKERS", "app"]
