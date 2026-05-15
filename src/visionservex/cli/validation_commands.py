# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""visionservex validation run — execute the test suite with a named profile."""

from __future__ import annotations

import json
import subprocess
import sys

import typer
from rich.console import Console

app = typer.Typer(help="Run the VisionServeX validation suite with a named profile.")
console = Console()

_PROFILES: dict[str, list[str]] = {
    "release": [
        "-q",
        "--tb=short",
        "-m",
        "not real_model and not gpu and not slow",
        "tests/",
    ],
    "local": [
        "-q",
        "--tb=short",
        "tests/",
    ],
    "gpu": [
        "-q",
        "--tb=short",
        "-m",
        "gpu or real_model",
        "tests/",
    ],
    "syntax": [
        "-q",
        "--tb=short",
        "tests/test_syntax_contract.py",
    ],
}

_ENV_FLAGS: dict[str, dict[str, str]] = {
    "gpu": {"VISION_SERVEX_RUN_GPU_TESTS": "1", "VISION_SERVEX_RUN_REAL_MODEL_TESTS": "1"},
    "local": {"VISION_SERVEX_RUN_REAL_MODEL_TESTS": "1"},
    "release": {},
    "syntax": {},
}


@app.command("run", help="Run the validation suite with a named profile.")
def run(
    profile: str = typer.Argument("release", help="Profile: release | local | gpu | syntax"),
    json_: bool = typer.Option(False, "--json"),
    extra: list[str] = typer.Argument(None),
) -> None:
    """Run pytest with the selected validation profile.

    Profiles:
      release  — standard CI (no real weights, no GPU)
      local    — full local tests including real_model
      gpu      — GPU-enabled tests (requires CUDA/MPS)
      syntax   — syntax contract tests only
    """
    if profile not in _PROFILES:
        msg = f"Unknown profile {profile!r}. Available: {', '.join(_PROFILES)}"
        if json_:
            typer.echo(json.dumps({"error": msg}))
        else:
            console.print(f"[red]{msg}[/red]")
        raise typer.Exit(1)

    pytest_args = _PROFILES[profile] + list(extra or [])
    env_extras = _ENV_FLAGS[profile]

    import os

    env = dict(os.environ)
    env.update(env_extras)
    env["PYTHONPATH"] = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "src")

    if not json_:
        console.print(f"[bold]Running validation profile: {profile}[/bold]")
        console.print(f"  pytest {' '.join(pytest_args)}")
        if env_extras:
            for k, v in env_extras.items():
                console.print(f"  env: {k}={v}")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", *pytest_args],
        env=env,
    )

    if json_:
        typer.echo(
            json.dumps(
                {
                    "profile": profile,
                    "pytest_args": pytest_args,
                    "return_code": result.returncode,
                    "passed": result.returncode == 0,
                },
                indent=2,
            )
        )
    elif result.returncode == 0:
        console.print(f"\n[green]✓ Validation profile '{profile}' passed.[/green]")
    else:
        console.print(f"\n[red]✗ Validation profile '{profile}' failed.[/red]")
        raise typer.Exit(result.returncode)


__all__ = ["app"]
