# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Florence-2 dedicated CLI — environment doctor and smoke-test.

Florence-2 requires transformers < 5.0.  Use the [florence2] extra in a
dedicated environment; this module helps users diagnose and verify.

Install recipe::

    conda create -n vsrv-florence python=3.11 -y
    conda activate vsrv-florence
    pip install "visionservex[florence2]"
    visionservex florence2 smoke-test florence-2-base image.jpg --task caption

Reference:
- https://github.com/microsoft/Florence-2
- https://huggingface.co/microsoft/Florence-2-base
- https://huggingface.co/microsoft/Florence-2-large
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="Florence-2 environment doctor and smoke-test (requires transformers < 5.0).",
    no_args_is_help=True,
)
console = Console()

_SUPPORTED_TASKS = [
    "caption",
    "detailed_caption",
    "more_detailed_caption",
    "object_detection",
    "dense_caption",
    "phrase_grounding",
    "ocr",
    "region_ocr",
]

_SETUP_RECIPE = """
  # Florence-2 requires a separate environment with transformers < 5.0:
  conda create -n vsrv-florence python=3.11 -y
  conda activate vsrv-florence
  pip install "visionservex[florence2]"
  # Then run:
  visionservex florence2 smoke-test florence-2-base <image> --task caption
"""


def _check_transformers_version() -> tuple[bool, str]:
    """Return (compatible, version_string). Compatible means < 5.0."""
    try:
        import transformers  # type: ignore

        ver = transformers.__version__
        major = int(ver.split(".")[0])
        return major < 5, ver
    except ImportError:
        return False, "NOT_INSTALLED"


@app.command("doctor")
def doctor(json_: bool = typer.Option(False, "--json")) -> None:
    """Check whether the current environment supports Florence-2."""
    compatible, tr_ver = _check_transformers_version()

    checks = {
        "transformers_version": tr_ver,
        "transformers_compatible": compatible,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
    }

    try:
        import torch  # type: ignore  # noqa: F401

        checks["torch"] = "installed"
    except ImportError:
        checks["torch"] = "NOT_INSTALLED"

    if compatible:
        checks["status"] = "ok"
        checks["message"] = "Environment is Florence-2 compatible."
        checks["next_step"] = (
            "visionservex florence2 smoke-test florence-2-base <image> --task caption"
        )
    else:
        checks["status"] = "FLORENCE2_TRANSFORMERS_VERSION_UNSUPPORTED"
        checks["message"] = (
            f"Florence-2 requires transformers>=4.40,<5.0; current env has transformers {tr_ver}"
        )
        checks["setup_recipe"] = _SETUP_RECIPE.strip()

    if json_:
        print(json.dumps(checks, indent=2))
        return

    table = Table(title="Florence-2 environment doctor", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for k, v in checks.items():
        table.add_row(k, str(v))
    console.print(table)

    if not compatible:
        console.print(f"\n[red]{checks['status']}[/red]")
        console.print(f"  {checks['message']}")
        console.print("\n[bold]Setup recipe:[/bold]")
        console.print(f"[dim]{_SETUP_RECIPE}[/dim]")
    else:
        console.print(f"\n[green]{checks['message']}[/green]")


@app.command("smoke-test")
def smoke_test(
    model_id: str = typer.Argument(..., help="Model ID: florence-2-base or florence-2-large."),
    image: Path = typer.Argument(..., help="Path to an image file."),
    task: str = typer.Option("caption", "--task", help=f"Task: {', '.join(_SUPPORTED_TASKS)}"),
    prompt: str = typer.Option("", "--prompt", help="Phrase prompt for phrase_grounding."),
    auto_pull: bool = typer.Option(False, "--auto-pull", help="Download checkpoint if missing."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Run a Florence-2 inference smoke test in the current environment.

    Fails with FLORENCE2_TRANSFORMERS_VERSION_UNSUPPORTED if transformers >= 5.0.
    """
    compatible, tr_ver = _check_transformers_version()

    if not compatible:
        payload = {
            "status": "FLORENCE2_TRANSFORMERS_VERSION_UNSUPPORTED",
            "transformers_version": tr_ver,
            "message": (
                f"Florence-2 requires transformers>=4.40,<5.0; "
                f"current env has transformers {tr_ver}"
            ),
            "setup_recipe": _SETUP_RECIPE.strip(),
        }
        if json_:
            print(json.dumps(payload, indent=2))
            return
        console.print(f"[red]{payload['status']}[/red]: {payload['message']}")
        console.print(f"\n[bold]Setup:[/bold]\n{_SETUP_RECIPE}")
        raise typer.Exit(3)

    if not image.exists():
        console.print(f"[red]Image not found:[/red] {image}")
        raise typer.Exit(2)

    if task not in _SUPPORTED_TASKS:
        console.print(f"[red]Unknown task {task!r}.[/red] Choose: {_SUPPORTED_TASKS}")
        raise typer.Exit(2)

    from PIL import Image as PILImage

    from visionservex import VisionModel

    m = VisionModel(model_id, auto_pull=auto_pull)
    img = PILImage.open(image).convert("RGB")

    kw: dict = {"task": task}
    if prompt:
        kw["prompt"] = prompt

    result = m.predict(img, **kw)
    payload = result.to_dict()
    payload["task"] = task
    payload["transformers_version"] = tr_ver

    if json_:
        print(json.dumps(payload, indent=2))
        return

    console.print(f"[bold]{result.summary()}[/bold]")
    text = result.metadata.get("text", "")
    if text:
        console.print(f"  text: {text[:200]}")
    for d in result.detections[:3]:
        console.print(f"  {d.label}: {d.score:.3f}")


@app.command("create-env")
def create_env(
    name: str = typer.Option("visionservex-florence", "--name", help="Conda environment name."),
    python: str = typer.Option("3.11", "--python", help="Python version."),
    execute: bool = typer.Option(
        False, "--execute", help="Actually run the commands (requires conda in PATH)."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Generate (or execute) the conda/pip recipe for a Florence-2 compatible environment.

    By default this prints the commands without running them.
    Pass --execute to attempt running them automatically.
    """
    import shutil
    import subprocess

    # transformers==4.46.3 is the validated version:
    # - passes check_imports for conditional flash_attn blocks
    # - does not have the _supports_sdpa AttributeError from 4.57.x
    # - confirmed smoke result: "a red truck with a light on top of it" on street.jpg
    commands = [
        f"conda create -n {name} python={python} -y",
        f"conda run -n {name} pip install -U pip",
        f'conda run -n {name} pip install "visionservex[florence2]"',
        f'conda run -n {name} pip install "transformers==4.46.3" einops timm accelerate',
        f"conda run -n {name} python -c \"import transformers; print('transformers version:', transformers.__version__)\"",
        f"conda run -n {name} visionservex florence2 doctor",
    ]
    payload = {
        "env_name": name,
        "python": python,
        "transformers_pin": "transformers==4.46.3",
        "required_extras": ["einops", "timm"],
        "commands": commands,
        "smoke_test_command": f"conda run -n {name} visionservex florence2 smoke-test florence-2-base <image> --task caption",
        "validated_smoke_result": "transformers 4.46.3 + Florence-2-base: caption PASSED (street.jpg → 'a red truck with a light on top of it')",
    }

    if json_:
        print(json.dumps(payload, indent=2))
        return

    console.print(f"[bold]Florence-2 environment setup: {name}[/bold]\n")
    for cmd in commands:
        console.print(f"  [cyan]{cmd}[/cyan]")
    console.print(f"\n[dim]Smoke test: {payload['smoke_test_command']}[/dim]")

    if not execute:
        console.print("\n[dim](Pass --execute to run these commands automatically)[/dim]")
        return

    if not shutil.which("conda"):
        console.print("[red]conda not found in PATH. Cannot --execute.[/red]")
        raise typer.Exit(2)

    for cmd in commands:
        console.print(f"\n[bold]Running:[/bold] {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, check=True)
            console.print(f"[green]✓ {result.returncode}[/green]")
        except subprocess.CalledProcessError as exc:
            console.print(f"[red]Failed:[/red] {exc}")
            raise typer.Exit(1) from exc

    console.print(
        f"\n[green]Environment '{name}' ready.[/green] "
        f"Run: conda activate {name} && {payload['smoke_test_command']}"
    )


__all__ = ["app"]
