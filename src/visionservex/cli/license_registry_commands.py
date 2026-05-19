# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.45.0: license-gate check and registry validate CLIs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import typer
from rich.console import Console

license_gate_app = typer.Typer(
    help="v2.45.0: license gate management for opt-in models.",
    no_args_is_help=True,
)
registry_app = typer.Typer(
    help="v2.45.0: model registry validation.",
    no_args_is_help=True,
)
console = Console()

# Models requiring explicit opt-in with their licenses
_LICENSE_REGISTRY: dict[str, dict] = {
    "fastsam-s": {
        "license": "AGPL-3.0",
        "flag": "--accept-agpl",
        "env_var": "VISIONSERVEX_ACCEPT_AGPL",
        "source": "https://github.com/CASIA-IVA-Lab/FastSAM",
    },
    "fastsam-x": {
        "license": "AGPL-3.0",
        "flag": "--accept-agpl",
        "env_var": "VISIONSERVEX_ACCEPT_AGPL",
        "source": "https://github.com/CASIA-IVA-Lab/FastSAM",
    },
    "yolo-world": {
        "license": "AGPL-3.0",
        "flag": "--accept-agpl",
        "env_var": "VISIONSERVEX_ACCEPT_AGPL",
        "source": "https://github.com/AILab-CVC/YOLO-World",
    },
    "yolo11x.pt": {
        "license": "AGPL-3.0",
        "flag": "--accept-agpl",
        "env_var": "VISIONSERVEX_ACCEPT_AGPL",
        "source": "https://github.com/ultralytics/ultralytics",
    },
    "yolo11l-seg.pt": {
        "license": "AGPL-3.0",
        "flag": "--accept-agpl",
        "env_var": "VISIONSERVEX_ACCEPT_AGPL",
        "source": "https://github.com/ultralytics/ultralytics",
    },
    "yolo11x-seg.pt": {
        "license": "AGPL-3.0",
        "flag": "--accept-agpl",
        "env_var": "VISIONSERVEX_ACCEPT_AGPL",
        "source": "https://github.com/ultralytics/ultralytics",
    },
    "yolo26x.pt": {
        "license": "AGPL-3.0",
        "flag": "--accept-agpl",
        "env_var": "VISIONSERVEX_ACCEPT_AGPL",
        "source": "https://github.com/ultralytics/ultralytics",
    },
    "yolo26x-seg.pt": {
        "license": "AGPL-3.0",
        "flag": "--accept-agpl",
        "env_var": "VISIONSERVEX_ACCEPT_AGPL",
        "source": "https://github.com/ultralytics/ultralytics",
    },
    "yolov10b.pt": {
        "license": "AGPL-3.0",
        "flag": "--accept-agpl",
        "env_var": "VISIONSERVEX_ACCEPT_AGPL",
        "source": "https://github.com/ultralytics/ultralytics",
    },
    "yolov8x.pt": {
        "license": "AGPL-3.0",
        "flag": "--accept-agpl",
        "env_var": "VISIONSERVEX_ACCEPT_AGPL",
        "source": "https://github.com/ultralytics/ultralytics",
    },
    "yolov8x-seg.pt": {
        "license": "AGPL-3.0",
        "flag": "--accept-agpl",
        "env_var": "VISIONSERVEX_ACCEPT_AGPL",
        "source": "https://github.com/ultralytics/ultralytics",
    },
    "rfdetr-seg-xlarge": {
        "license": "PML-1.0",
        "flag": "--accept-pml",
        "env_var": "VISIONSERVEX_ACCEPT_PML",
        "source": "https://github.com/roboflow/rf-detr",
    },
    "rfdetr-seg-2xlarge": {
        "license": "PML-1.0",
        "flag": "--accept-pml",
        "env_var": "VISIONSERVEX_ACCEPT_PML",
        "source": "https://github.com/roboflow/rf-detr",
    },
    "prithvi-eo-2.0": {
        "license": "Apache-2.0-non-core",
        "flag": "--accept-non-core-license",
        "env_var": "VISIONSERVEX_ACCEPT_NON_CORE_LICENSE",
        "source": "https://github.com/NASA-IMPACT/Prithvi-EO",
    },
    "totalsegmentator": {
        "license": "Apache-2.0-non-core",
        "flag": "--accept-non-core-license",
        "env_var": "VISIONSERVEX_ACCEPT_NON_CORE_LICENSE",
        "source": "https://github.com/wasserth/TotalSegmentator",
    },
}

# Registry state overrides
_REGISTRY_STATES: dict[str, dict] = {
    "agriclip": {
        "state": "not_advertised",
        "reason": "AgriCLIP is audit-only; not in official benchmark suite.",
        "can_benchmark": False,
    },
    "deim-m": {
        "state": "upstream_deprecated",
        "reason": "DEIM v1 deprecated upstream by authors. Use DEIMv2 family instead.",
        "replacement": "deimv2-m",
        "can_benchmark": False,
    },
    "deim-s": {
        "state": "upstream_deprecated",
        "reason": "DEIM v1 deprecated upstream by authors. Use DEIMv2 family instead.",
        "replacement": "deimv2-s",
        "can_benchmark": False,
    },
    "dinov3-vitb16": {
        "state": "not_advertised",
        "reason": "DINOv3 vit-b/16 awaits official open weight publication.",
        "can_benchmark": False,
    },
    "oneformer-convnext-large": {
        "state": "wrong_registry_entry",
        "reason": "SHI-Labs trained OneFormer ConvNeXt on Cityscapes/ADE20K, NOT COCO. Use shi-labs/oneformer_coco_swin_large or oneformer_coco_dinat_large.",
        "replacement": "oneformer-dinat-large",
        "can_benchmark": False,
    },
}


def _emit(payload: dict, *, out: Path | None, fmt: str) -> None:
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2))
    else:
        color = "green" if payload.get("status") == "ok" else "yellow"
        console.print(f"[{color}]{payload.get('code', '?')}[/{color}]")
        for k in ("model_id", "license", "opt_in_required", "opt_in_command", "default_safe"):
            if k in payload:
                console.print(f"  {k}: {payload[k]}")


@license_gate_app.command("check")
def license_gate_check_cmd(
    model_id: str = typer.Argument(..., help="Model ID to check license gate for."),
    accept_agpl: bool = typer.Option(
        False, "--accept-agpl", help="Explicitly accept AGPL-3.0 license."
    ),
    accept_pml: bool = typer.Option(
        False, "--accept-pml", help="Explicitly accept PML-1.0 Plus license."
    ),
    accept_non_core: bool = typer.Option(
        False, "--accept-non-core-license", help="Accept non-core optional license."
    ),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """v2.45.0: check whether a model requires license opt-in to run."""
    info = _LICENSE_REGISTRY.get(model_id)
    if info is None:
        _emit(
            {
                "status": "ok",
                "code": "DEFAULT_SAFE",
                "model_id": model_id,
                "license": "default_safe",
                "opt_in_required": False,
                "default_safe": True,
                "message": f"{model_id} does not require explicit license opt-in.",
            },
            out=out,
            fmt=fmt,
        )
        return

    license_str = info["license"]
    env_var = info["env_var"]
    env_accepted = os.environ.get(env_var, "").strip() in ("1", "true", "yes")
    flag_accepted = (
        ("AGPL" in license_str and accept_agpl)
        or ("PML" in license_str and accept_pml)
        or ("non-core" in license_str and accept_non_core)
    )
    opted_in = env_accepted or flag_accepted

    payload = {
        "status": "ok" if opted_in else "license_gate",
        "code": "OPT_IN_ACCEPTED" if opted_in else "LICENSE_GATE_NOT_PASSED",
        "model_id": model_id,
        "license": license_str,
        "license_source_url": info.get("source", ""),
        "opt_in_required": not opted_in,
        "default_safe": False,
        "opt_in_env_var": env_var,
        "opt_in_cli_flag": info["flag"],
        "opt_in_command": f"VISIONSERVEX_ACCEPT_AGPL=1 visionservex predict {model_id} IMAGE"
        if "AGPL" in license_str
        else f"{info['flag']} visionservex predict {model_id} IMAGE",
        "excluded_from_default_safe_leaderboard": True,
        "message": (
            f"License opt-in accepted for {model_id} ({license_str})."
            if opted_in
            else f"{model_id} requires license opt-in. Set {env_var}=1 or pass {info['flag']}."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _emit(payload, out=out, fmt=fmt)
    if not opted_in:
        raise typer.Exit(1)


@registry_app.command("validate")
def registry_validate_cmd(
    model_id: str = typer.Argument(..., help="Model ID to validate in registry."),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """v2.45.0: validate registry entry for a model."""
    override = _REGISTRY_STATES.get(model_id)
    if override:
        payload = {
            "status": "ok",
            "code": override["state"].upper(),
            "model_id": model_id,
            "final_state": override["state"],
            "reason": override["reason"],
            "can_benchmark": override.get("can_benchmark", False),
            "replacement": override.get("replacement", ""),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    else:
        # Try to look up in SOURCE_MANIFEST
        try:
            from visionservex.model_zoo.manifest import SOURCE_MANIFEST

            src = SOURCE_MANIFEST.get(model_id)
            if src:
                payload = {
                    "status": "ok",
                    "code": "REGISTERED",
                    "model_id": model_id,
                    "family": src.family,
                    "task": src.task,
                    "license": src.license,
                    "runnable_in_visionservex": src.runnable_in_visionservex,
                    "recommended_action": src.recommended_action,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            else:
                payload = {
                    "status": "ok",
                    "code": "NOT_IN_MANIFEST",
                    "model_id": model_id,
                    "message": f"{model_id} is not in the VisionServeX model manifest.",
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
        except Exception as e:
            payload = {
                "status": "failed",
                "code": "MANIFEST_ERROR",
                "error": str(e),
                "model_id": model_id,
            }

    _emit(payload, out=out, fmt=fmt)


__all__ = ["license_gate_app", "registry_app"]
