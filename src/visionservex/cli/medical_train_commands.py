# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Medical training-truth CLI: `visionservex medical train ...`.

Honest by construction: VisionServeX does not train/fine-tune medical models
in-process. These commands expose the capability matrix, validate datasets, and
generate the EXACT upstream training commands (dry-run) — nothing more.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(
    help="Medical training truth: matrix, dataset validation, dry-run.", no_args_is_help=True
)


def _emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2, default=str))


@app.command("doctor")
def doctor(json_: bool = typer.Option(False, "--json")) -> None:
    """Show the medical training capability matrix + framework presence."""
    from visionservex.medical.training import train_doctor

    _emit(train_doctor())


@app.command("validate-dataset")
def validate_dataset(
    dataset_dir: Path = typer.Argument(..., help="Dataset root with images/ and masks/."),
    task: str = typer.Option("segmentation", "--task"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Validate a 2D segmentation dataset (image/mask pairing + shapes)."""
    from visionservex.medical.training import validate_segmentation_dataset

    report = validate_segmentation_dataset(dataset_dir, task=task)
    _emit(report)
    if not report["valid"]:
        raise typer.Exit(2)


@app.command("dry-run")
def dry_run_cmd(
    framework: str = typer.Option(..., "--framework", help="nnunet | monai | medsam2"),
    dataset_dir: Path = typer.Option(..., "--dataset"),
    out: Path = typer.Option(..., "--out"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Validate a dataset and emit the exact upstream training command (no training)."""
    from visionservex.medical.training import dry_run

    payload = dry_run(framework, dataset_dir, out)
    _emit(payload)
    if payload.get("status") == "failed":
        raise typer.Exit(2)


__all__ = ["app"]
