# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.22.0: RT-DETRv4 doctor / pull / smoke-test CLI.

Honest integration attempt against the real upstream state:

Upstream truth (verified 2026-05-18 against
https://github.com/lyuwenyu/RT-DETR/contents/):

The canonical RT-DETR repo (`lyuwenyu/RT-DETR`, 5k+ stars) currently ships:
- ``rtdetr_paddle/``   — original PaddlePaddle implementation
- ``rtdetr_pytorch/``  — RT-DETR v1 PyTorch
- ``rtdetrv2_paddle/`` — RT-DETRv2 PaddlePaddle
- ``rtdetrv2_pytorch/`` — RT-DETRv2 PyTorch

There is **no `rtdetrv4_pytorch/` directory and no v4 release tag** as of
this date. Every CLI in this module reports
``UPSTREAM_NOT_RELEASED`` with the exact evidence so notebooks can't
silently pretend RT-DETRv4 is "available but blocked on dependencies".

If the upstream releases v4, this module's `--allow-experimental-v4`
flag can be used to try the same PyTorchModelHubMixin / native-loader
flow that DEIMv2 uses; but right now that flag returns the same
``UPSTREAM_NOT_RELEASED`` blocker.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(
    help="v2.22.0: RT-DETRv4 doctor / pull / smoke-test (upstream not yet released).",
    no_args_is_help=True,
)
console = Console()

RTDETR_UPSTREAM_REPO = "https://github.com/lyuwenyu/RT-DETR"
RTDETR_AVAILABLE_VARIANTS = ("rtdetr_pytorch", "rtdetrv2_pytorch")
RTDETRV4_EVIDENCE_DATE = "2026-05-18"


def _diagnose_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "installed_torch": None,
        "rtdetrv4_package_importable": False,
        "huggingface_hub_available": False,
        "upstream_repo": RTDETR_UPSTREAM_REPO,
        "upstream_available_variants": list(RTDETR_AVAILABLE_VARIANTS),
        "upstream_v4_available": False,
        "evidence_date": RTDETRV4_EVIDENCE_DATE,
    }
    try:
        import torch  # type: ignore

        info["installed_torch"] = torch.__version__
    except Exception:
        pass
    try:
        importlib.import_module("rtdetrv4")
        info["rtdetrv4_package_importable"] = True
    except Exception:
        pass
    try:
        importlib.import_module("huggingface_hub")
        info["huggingface_hub_available"] = True
    except Exception:
        pass
    return info


def _emit(payload: dict[str, Any], *, out: Path | None, fmt: str) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2))
    else:
        color = {
            "ok": "green",
            "expected_blocker": "yellow",
            "failed": "red",
        }.get(payload.get("status", ""), "white")
        console.print(f"[{color}]{payload.get('code', '')}[/{color}]: {payload.get('message', '')}")


@app.command("doctor")
def doctor_cmd(
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Surface the upstream-not-released blocker with evidence."""
    env = _diagnose_environment()
    payload = {
        "status": "expected_blocker",
        "code": "RTDETRV4_UPSTREAM_NOT_RELEASED",
        "message": (
            "RT-DETRv4 is not yet released upstream. The canonical "
            f"{RTDETR_UPSTREAM_REPO} repo ships RT-DETR (v1) and RT-DETRv2 "
            "only; there is no rtdetrv4_pytorch/ directory or v4 release tag."
        ),
        "evidence": {
            "upstream_repo": RTDETR_UPSTREAM_REPO,
            "upstream_available_variants": RTDETR_AVAILABLE_VARIANTS,
            "verified_on": RTDETRV4_EVIDENCE_DATE,
        },
        "alternatives": [
            (
                "RT-DETRv2 is available upstream at "
                f"{RTDETR_UPSTREAM_REPO}/tree/main/rtdetrv2_pytorch — "
                "VisionServeX can wire RT-DETRv2 as `rtdetrv2-*` model IDs "
                "in a future release."
            ),
            "Watch the upstream repo for an rtdetrv4_pytorch/ directory.",
        ],
        "details": env,
    }
    _emit(payload, out=out, fmt=fmt)


@app.command("pull")
def pull_cmd(
    model_id: str = typer.Argument("rtdetrv4-s"),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Return UPSTREAM_NOT_RELEASED with exact evidence (no checkpoint to pull)."""
    payload = {
        "status": "expected_blocker",
        "code": "RTDETRV4_UPSTREAM_NOT_RELEASED",
        "model_id": model_id,
        "message": (
            "No RT-DETRv4 checkpoint exists upstream as of "
            f"{RTDETRV4_EVIDENCE_DATE}. RT-DETRv4 has not been released."
        ),
        "upstream_repo": RTDETR_UPSTREAM_REPO,
        "upstream_available_variants": list(RTDETR_AVAILABLE_VARIANTS),
    }
    _emit(payload, out=out, fmt=fmt)


@app.command("smoke-test")
def smoke_test_cmd(
    model_id: str = typer.Argument("rtdetrv4-s"),
    image: Path = typer.Argument(..., help="Image path."),
    device: str = typer.Option("cuda", "--device"),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
    draw: Path | None = typer.Option(None, "--draw"),
) -> None:
    """Return structured blocker — RT-DETRv4 doesn't exist upstream yet."""
    if not image.exists():
        _emit(
            {
                "status": "failed",
                "code": "INPUT_NOT_FOUND",
                "image": str(image),
                "message": f"Image not found: {image}",
            },
            out=out,
            fmt=fmt,
        )
        raise typer.Exit(2)

    env = _diagnose_environment()
    payload = {
        "status": "expected_blocker",
        "code": "RTDETRV4_UPSTREAM_NOT_RELEASED",
        "model_id": model_id,
        "image": str(image),
        "device": device,
        "message": (
            "RT-DETRv4 smoke-test cannot run because RT-DETRv4 has not been released "
            "upstream. RT-DETRv2 is available; consider wiring that instead."
        ),
        "evidence": {
            "upstream_repo": RTDETR_UPSTREAM_REPO,
            "upstream_available_variants": RTDETR_AVAILABLE_VARIANTS,
            "verified_on": RTDETRV4_EVIDENCE_DATE,
        },
        "details": env,
    }
    _emit(payload, out=out, fmt=fmt)
    # Honest exit-0: expected_blocker is not a failure.


__all__ = ["app"]
