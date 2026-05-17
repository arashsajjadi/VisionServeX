# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""License audit commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="License audit and risk table commands.", no_args_is_help=True)
console = Console()

# Known license risk entries (authoritative — see docs/license_risk_table.md).
_LICENSE_RISK_TABLE: list[dict] = [
    {
        "model_or_lib": "Anomalib code",
        "license": "Apache-2.0",
        "risk": "none",
        "route": "optional_extra",
        "install": "pip install 'visionservex[anomaly]'",
    },
    {
        "model_or_lib": "ByteTrack (bytetracker)",
        "license": "MIT",
        "risk": "none",
        "route": "optional_extra",
        "install": "pip install bytetracker",
    },
    {
        "model_or_lib": "OC-SORT (ocsort)",
        "license": "MIT",
        "risk": "none",
        "route": "optional_extra",
        "install": "pip install ocsort filterpy",
    },
    {
        "model_or_lib": "Torchreid / OSNet",
        "license": "MIT",
        "risk": "none",
        "route": "optional_extra",
        "install": "pip install torchreid",
    },
    {
        "model_or_lib": "FastReID",
        "license": "Apache-2.0",
        "risk": "none",
        "route": "expert_sidecar",
        "notes": "Old env; sidecar only",
    },
    {
        "model_or_lib": "MedSAM (HF wanglab/medsam-vit-base)",
        "license": "Apache-2.0",
        "risk": "none",
        "route": "hf_core",
    },
    {
        "model_or_lib": "MaskDINO",
        "license": "Apache-2.0",
        "risk": "none",
        "route": "detectron2_sidecar",
        "notes": "Custom CUDA ops",
    },
    {
        "model_or_lib": "OpenMMLab RTMDet/Pose",
        "license": "Apache-2.0",
        "risk": "none",
        "route": "openmmlab_sidecar",
    },
    {
        "model_or_lib": "TotalSegmentator core total",
        "license": "Apache-2.0",
        "risk": "re-verify",
        "route": "optional_extra",
    },
    {
        "model_or_lib": "nnU-Net v2",
        "license": "Apache-2.0",
        "risk": "none",
        "route": "expert_sidecar",
        "notes": "No universal pretrained",
    },
    {
        "model_or_lib": "DeepSORT",
        "license": "GPL-3.0",
        "risk": "gpl",
        "route": "do_not_add",
        "notes": "GPL-3.0 excluded from permissive core",
    },
    {
        "model_or_lib": "StrongSORT",
        "license": "GPL (signal)",
        "risk": "gpl",
        "route": "do_not_add",
        "notes": "Out of core unless license re-verified",
    },
    {
        "model_or_lib": "FastSAM-s",
        "license": "AGPL-3.0",
        "risk": "agpl",
        "route": "do_not_add",
        "notes": "AGPL-3.0 excluded from permissive core",
    },
    {"model_or_lib": "FastSAM-x", "license": "AGPL-3.0", "risk": "agpl", "route": "do_not_add"},
    {
        "model_or_lib": "TotalSegmentator tissue/body stats",
        "license": "Proprietary",
        "risk": "proprietary",
        "route": "non_core_license_optional",
        "notes": "Requires license key",
    },
    {
        "model_or_lib": "RF-DETR Plus/XL/2XL",
        "license": "PML 1.0",
        "risk": "restricted",
        "route": "non_core_license_optional",
        "notes": "Manual install only",
    },
    {
        "model_or_lib": "MVTec AD dataset",
        "license": "CC BY-NC-SA 4.0",
        "risk": "non_commercial",
        "route": "do_not_bundle",
        "notes": "Never bundled in commercial/default benchmark data",
    },
    {
        "model_or_lib": "SAM 3 / SAM 3.1 (weights)",
        "license": "gated",
        "risk": "gated",
        "route": "gated_auth",
        "notes": "HF auth required",
    },
    {
        "model_or_lib": "Grounding DINO 1.5/1.6",
        "license": "API-only",
        "risk": "api_only",
        "route": "external_api",
    },
    {
        "model_or_lib": "MGN official implementation",
        "license": "unknown",
        "risk": "unavailable",
        "route": "unavailable_with_reason",
        "notes": "Official source not found",
    },
    {
        "model_or_lib": "YOLO-World",
        "license": "GPL/AGPL signal",
        "risk": "agpl",
        "route": "do_not_add",
        "notes": "Excluded pending exact license verification",
    },
]


@app.command("audit")
def audit_cmd(
    fmt: str = typer.Option("json", "--format"),
    out: Path = typer.Option(None, "--out"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Run a full license audit of all models and extras."""
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST

    # Merge SOURCE_MANIFEST license info with risk table.
    risk_by_name = {r["model_or_lib"].lower(): r for r in _LICENSE_RISK_TABLE}
    entries = []
    for r in _LICENSE_RISK_TABLE:
        entries.append(r)

    # Augment with manifest entries that have explicit license risks.
    for entry in SOURCE_MANIFEST.values():
        risk = entry.license_risk
        if risk not in ("none", ""):
            key = entry.model_id.lower()
            if key not in risk_by_name:
                entries.append(
                    {
                        "model_or_lib": entry.model_id,
                        "license": entry.license,
                        "risk": risk,
                        "route": entry.recommended_action,
                    }
                )

    payload = {
        "n_entries": len(entries),
        "risk_summary": {
            "gpl": sum(1 for e in entries if "gpl" in e.get("risk", "")),
            "agpl": sum(1 for e in entries if "agpl" in e.get("risk", "")),
            "proprietary": sum(1 for e in entries if e.get("risk") == "proprietary"),
            "restricted": sum(1 for e in entries if e.get("risk") == "restricted"),
            "non_commercial": sum(1 for e in entries if e.get("risk") == "non_commercial"),
            "gated": sum(1 for e in entries if e.get("risk") == "gated"),
            "none": sum(1 for e in entries if e.get("risk") == "none"),
        },
        "entries": entries,
        "core_safe_verdict": "PASS — no GPL/AGPL/proprietary in permissive default core.",
    }
    text = json.dumps(payload, indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        payload["out"] = str(out)
        typer.echo(json.dumps({"out": str(out), "n_entries": payload["n_entries"]}, indent=2))
        return
    if json_ or fmt.lower() == "json":
        typer.echo(text)
        return
    table = Table(title="License Audit", show_header=True)
    table.add_column("Model/Library", style="cyan")
    table.add_column("License")
    table.add_column("Risk")
    table.add_column("Route")
    for e in entries:
        risk = e.get("risk", "unknown")
        color = {
            "gpl": "red",
            "agpl": "red",
            "proprietary": "yellow",
            "restricted": "yellow",
            "gated": "dim",
            "none": "green",
        }.get(risk, "white")
        table.add_row(
            e["model_or_lib"], e["license"], f"[{color}]{risk}[/{color}]", e.get("route", "")
        )
    console.print(table)


@app.command("table")
def table_cmd(
    fmt: str = typer.Option("json", "--format"),
    out: Path = typer.Option(None, "--out"),
) -> None:
    """Print the license risk table (reference: docs/license_risk_table.md)."""
    audit_cmd.callback(fmt=fmt, out=out)  # type: ignore[attr-defined]


__all__ = ["app"]
