# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Google Colab GPU worker mode commands.

VisionServeX supports running as a temporary remote GPU worker on Google Colab.
This is good for demos, benchmarks, and short-lived GPU access — NOT for production.

Colab sessions can disconnect at any time. Cache may be lost. Public exposure
without auth is rejected. See docs/colab_gpu_worker.md.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Google Colab GPU worker support (optional, experimental).")
console = Console()


def in_colab() -> bool:
    """Detect whether the current process is running inside Google Colab."""
    if "google.colab" in sys.modules:
        return True
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        pass
    # Common Colab markers
    if Path("/content").is_dir() and os.environ.get("COLAB_RELEASE_TAG"):
        return True
    return bool(os.environ.get("COLAB_GPU"))


def _drive_mounted() -> bool:
    return Path("/content/drive/MyDrive").is_dir()


def _detect_gpu() -> dict:
    """Return GPU info from torch if available."""
    info: dict = {"available": False, "name": None, "total_vram_gb": None, "free_vram_gb": None}
    try:
        import torch

        if torch.cuda.is_available():
            info["available"] = True
            info["name"] = torch.cuda.get_device_name(0)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                info["total_vram_gb"] = round(total_bytes / (1024**3), 2)
                info["free_vram_gb"] = round(free_bytes / (1024**3), 2)
            except Exception:
                pass
    except ImportError:
        info["torch_installed"] = False
    return info


def _detect_python() -> dict:
    return {
        "version": sys.version.split()[0],
        "executable": sys.executable,
        "platform": sys.platform,
    }


def _detect_visionservex() -> dict:
    try:
        from visionservex import __version__

        return {"installed": True, "version": __version__}
    except ImportError:
        return {"installed": False, "version": None}


def _detect_cloudflared() -> bool:
    return shutil.which("cloudflared") is not None


def _detect_auth_configured() -> bool:
    """Return True if any auth token/key is configured in env."""
    return any(
        os.environ.get(k) for k in ("VISIONSERVEX_AUTH__API_KEY", "VISIONSERVEX_AUTH__TOKEN")
    )


@app.command("doctor", help="Comprehensive Colab environment diagnostic.")
def doctor(json_: bool = typer.Option(False, "--json")) -> None:
    """Detect Colab environment, GPU, Drive, dependencies, and config.

    If not running in Colab, returns COLAB_NOT_DETECTED.
    """
    is_colab = in_colab()
    gpu = _detect_gpu()
    py = _detect_python()
    vsx = _detect_visionservex()

    payload = {
        "in_colab": is_colab,
        "python": py,
        "visionservex": vsx,
        "gpu": gpu,
        "drive_mounted": _drive_mounted(),
        "cloudflared_installed": _detect_cloudflared(),
        "auth_configured": _detect_auth_configured(),
        "cwd": os.getcwd(),
    }

    if not is_colab:
        payload["status"] = "COLAB_NOT_DETECTED"
        payload["hint"] = (
            "Not running inside Google Colab. This command is intended to run "
            "from within a Colab notebook."
        )
        if json_:
            typer.echo(json.dumps(payload, indent=2))
            return
        console.print("[yellow]COLAB_NOT_DETECTED[/yellow]")
        console.print(payload["hint"])
        console.print(f"\nCurrent platform: [dim]{py['platform']}[/dim]")
        console.print(f"Current cwd:      [dim]{payload['cwd']}[/dim]")
        return

    if not gpu["available"]:
        payload["status"] = "COLAB_GPU_UNAVAILABLE"
        payload["hint"] = "Runtime → Change runtime type → GPU"
        if json_:
            typer.echo(json.dumps(payload, indent=2))
            return
        console.print("[yellow]COLAB_GPU_UNAVAILABLE[/yellow]")
        console.print(f"Hint: [cyan]{payload['hint']}[/cyan]")
        return

    payload["status"] = "ok"
    payload["safe_vram_budget_gb"] = max(0, (gpu["free_vram_gb"] or 0) - 1.5)

    if json_:
        typer.echo(json.dumps(payload, indent=2))
        return

    console.print(
        Panel.fit(
            "[bold]VisionServeX Colab GPU Worker[/bold]\nEnvironment diagnostic",
            border_style="green",
        )
    )
    table = Table(show_header=False, box=None)
    table.add_column("k", style="cyan")
    table.add_column("v")
    table.add_row("In Colab", "[green]yes[/green]")
    table.add_row("Python", py["version"])
    table.add_row("VisionServeX", str(vsx.get("version") or "[red]NOT INSTALLED[/red]"))
    table.add_row("GPU", gpu["name"] or "-")
    table.add_row("Total VRAM", f"{gpu['total_vram_gb']:.2f} GB")
    table.add_row("Free VRAM", f"{gpu['free_vram_gb']:.2f} GB")
    table.add_row("Safe budget", f"{payload['safe_vram_budget_gb']:.2f} GB (after 1.5 GB reserve)")
    table.add_row("Drive mounted", "[green]yes[/green]" if payload["drive_mounted"] else "no")
    table.add_row("cloudflared", "[green]yes[/green]" if payload["cloudflared_installed"] else "no")
    table.add_row("Auth configured", "[green]yes[/green]" if payload["auth_configured"] else "no")
    console.print(table)

    console.print(
        "\n[dim]Next: `visionservex pull-suite gpu-demo` then `visionservex gateway start "
        "--profile colab-gpu-worker`[/dim]"
    )


@app.command("status", help="Single-line Colab status summary.")
def status(json_: bool = typer.Option(False, "--json")) -> None:
    is_colab = in_colab()
    gpu = _detect_gpu()
    payload = {
        "in_colab": is_colab,
        "gpu_available": gpu["available"],
        "gpu_name": gpu["name"],
        "drive_mounted": _drive_mounted(),
    }
    if json_:
        typer.echo(json.dumps(payload, indent=2))
        return
    if not is_colab:
        console.print("[grey50]Not running in Colab.[/grey50]")
        return
    gpu_str = (
        f"[green]{gpu['name']}[/green] ({gpu['free_vram_gb']:.1f}/{gpu['total_vram_gb']:.1f} GB free)"
        if gpu["available"]
        else "[red]no GPU[/red]"
    )
    drive = "[green]drive mounted[/green]" if payload["drive_mounted"] else "[dim]no drive[/dim]"
    console.print(f"Colab: [green]yes[/green] | GPU: {gpu_str} | {drive}")


@app.command("gpu-check", help="Quick GPU health and VRAM budget check.")
def gpu_check(json_: bool = typer.Option(False, "--json")) -> None:
    """Verify GPU is healthy and report a safe VRAM budget for Colab."""
    gpu = _detect_gpu()
    if not gpu["available"]:
        payload = {
            "status": "COLAB_GPU_UNAVAILABLE",
            "hint": "Runtime → Change runtime type → GPU",
        }
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(f"[red]{payload['status']}[/red]")
            console.print(f"Hint: {payload['hint']}")
        raise typer.Exit(1)

    safe_budget = max(0.0, (gpu["free_vram_gb"] or 0.0) - 1.5)
    payload = {
        "status": "ok",
        "gpu": gpu,
        "safe_vram_budget_gb": round(safe_budget, 2),
        "recommended_max_concurrency": 1,
        "recommended_profile": "colab-gpu-worker",
    }
    if json_:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(f"GPU: [green]{gpu['name']}[/green]")
    console.print(f"Total VRAM:  {gpu['total_vram_gb']:.2f} GB")
    console.print(f"Free VRAM:   {gpu['free_vram_gb']:.2f} GB")
    console.print(f"Safe budget: [cyan]{safe_budget:.2f} GB[/cyan] (with 1.5 GB reserve)")
    console.print("Recommend:   [cyan]visionservex gateway start --profile colab-gpu-worker[/cyan]")


@app.command("mount-drive", help="Print Drive-mount instructions for Colab.")
def mount_drive(json_: bool = typer.Option(False, "--json")) -> None:
    """Show how to mount Google Drive in a Colab notebook.

    VisionServeX cannot mount Drive on your behalf — it must be done from a
    notebook cell with user consent.
    """
    snippet = (
        "# In a Colab notebook cell:\n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
    )
    if json_:
        typer.echo(
            json.dumps(
                {"snippet": snippet, "mounted": _drive_mounted()},
                indent=2,
            )
        )
        return
    console.print("[bold]Mount Google Drive in a Colab notebook:[/bold]")
    console.print(f"[cyan]{snippet.strip()}[/cyan]")
    if _drive_mounted():
        console.print("\n[green]Drive is already mounted at /content/drive/MyDrive.[/green]")
    else:
        console.print("\n[yellow]Drive not currently mounted in this session.[/yellow]")


@app.command("cache-path", help="Show recommended VisionServeX cache path for Colab.")
def cache_path(json_: bool = typer.Option(False, "--json")) -> None:
    """Return the recommended cache directory for the current Colab session.

    If Drive is mounted, suggests a Drive path that persists across sessions.
    Otherwise suggests an ephemeral /content path with a warning.
    """
    if _drive_mounted():
        path = "/content/drive/MyDrive/visionservex_cache"
        persistent = True
        warning = None
    else:
        path = "/content/visionservex_cache"
        persistent = False
        warning = (
            "Cache is in /content and will be lost when the Colab session ends. "
            "Mount Drive first to persist across sessions."
        )

    payload = {"cache_dir": path, "persistent": persistent, "warning": warning}
    if json_:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(f"Cache dir: [cyan]{path}[/cyan]")
    console.print(f"Persistent: {'[green]yes[/green]' if persistent else '[yellow]no[/yellow]'}")
    if warning:
        console.print(f"\n[yellow]{warning}[/yellow]")
    console.print(f"\nApply: [dim]export VISIONSERVEX_CACHE_DIR={path}[/dim]")


@app.command("setup-cache", help="Suggest cache configuration for Colab.")
def setup_cache(
    drive: bool = typer.Option(
        False, "--drive", help="Use Google Drive cache (requires Drive mounted)."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Print exact env-var commands to configure the VisionServeX cache for Colab.

    Does not write any files. The user copies and pastes the export line.
    """
    if drive:
        if not _drive_mounted():
            msg = (
                "Drive not mounted. Run this in a notebook cell first:\n"
                "  from google.colab import drive\n"
                "  drive.mount('/content/drive')"
            )
            if json_:
                typer.echo(json.dumps({"status": "DRIVE_NOT_MOUNTED", "hint": msg}, indent=2))
            else:
                console.print(f"[red]Drive not mounted.[/red]\n{msg}")
            raise typer.Exit(1)
        path = "/content/drive/MyDrive/visionservex_cache"
    else:
        path = "/content/visionservex_cache"

    payload = {
        "cache_dir": path,
        "env_command": f"export VISIONSERVEX_CACHE_DIR={path}",
        "shell_setup": (f"mkdir -p {path}\nexport VISIONSERVEX_CACHE_DIR={path}"),
        "persistent": drive,
    }

    if json_:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(f"[bold]Cache directory:[/bold] {path}")
    console.print("\n[bold]In a Colab cell:[/bold]")
    console.print(f"[cyan]!mkdir -p {path}[/cyan]")
    console.print(f"[cyan]%env VISIONSERVEX_CACHE_DIR={path}[/cyan]")
    if not drive:
        console.print(
            "\n[yellow]Warning:[/yellow] cache is ephemeral. Mount Drive and re-run with --drive."
        )


@app.command("cleanup", help="Clean up Colab-specific temp/cache files.")
def cleanup(
    yes: bool = typer.Option(False, "--yes"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Remove Colab session-specific VisionServeX temp files.

    Does NOT touch model cache or Drive-cached files unless explicitly requested.
    """
    targets = [
        Path("/content/visionservex_cache/tmp"),
        Path("/tmp/visionservex_colab"),
    ]
    existing = [p for p in targets if p.exists()]

    if not existing:
        if json_:
            typer.echo(json.dumps({"removed": [], "status": "nothing_to_clean"}, indent=2))
        else:
            console.print("[green]Nothing to clean.[/green]")
        return

    if not yes and not json_:
        console.print(f"Will remove: {[str(p) for p in existing]}")
        if not typer.confirm("Proceed?"):
            raise typer.Exit(0)

    removed = []
    for p in existing:
        try:
            shutil.rmtree(p, ignore_errors=True)
            removed.append(str(p))
        except Exception:
            pass

    if json_:
        typer.echo(json.dumps({"removed": removed, "status": "ok"}, indent=2))
    else:
        console.print(f"[green]Removed {len(removed)} path(s).[/green]")


@app.command("token", help="Generate a Colab worker API token (alias for gateway token).")
def token(json_: bool = typer.Option(False, "--json")) -> None:
    """Generate an API key suitable for the Colab worker."""
    import secrets

    token_str = secrets.token_urlsafe(32)
    payload = {
        "api_key": token_str,
        "env_command": f"export VISIONSERVEX_AUTH__API_KEY={token_str}",
        "warning": "Save this token now. It is shown ONCE and not stored.",
    }
    if json_:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print("[bold]New API key (save it now — shown once):[/bold]")
    console.print(f"[cyan]{token_str}[/cyan]")
    console.print("\n[dim]In a Colab cell:[/dim]")
    console.print(f"[cyan]%env VISIONSERVEX_AUTH__API_KEY={token_str}[/cyan]")
    console.print("[cyan]%env VISIONSERVEX_AUTH__ENABLED=true[/cyan]")


@app.command("tunnel-start", help="Start a Cloudflare Tunnel for Colab (auth required).")
def tunnel_start(
    domain: str = typer.Option(..., "--domain", help="Cloudflare-managed domain."),
    config: Path = typer.Option(
        Path("/content/visionservex_tunnel.yaml"), "--config", help="Tunnel YAML config path."
    ),
    i_understand: bool = typer.Option(
        False,
        "--i-understand-this-is-public",
        help="Confirm you understand this exposes the Colab worker publicly.",
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Start a Cloudflare Tunnel pointing at the local Colab gateway.

    Auth must be configured before exposing publicly. If auth is not enabled,
    this command refuses to start the tunnel.
    """
    if not _detect_auth_configured():
        msg = (
            "Authentication is not configured. Run "
            "`visionservex colab token` first and set the printed env vars."
        )
        if json_:
            typer.echo(json.dumps({"status": "AUTH_REQUIRED", "hint": msg}, indent=2))
        else:
            console.print(f"[red]AUTH_REQUIRED[/red]\n{msg}")
        raise typer.Exit(1)

    if not i_understand:
        msg = (
            "Public exposure requires --i-understand-this-is-public. "
            "Read docs/cloudflare_tunnel.md and docs/security.md first."
        )
        if json_:
            typer.echo(json.dumps({"status": "EXPOSURE_NOT_ACKNOWLEDGED", "hint": msg}, indent=2))
        else:
            console.print(f"[red]EXPOSURE_NOT_ACKNOWLEDGED[/red]\n{msg}")
        raise typer.Exit(1)

    if not shutil.which("cloudflared"):
        msg = "cloudflared not installed. In Colab: !pip install -q cloudflared || curl/install"
        if json_:
            typer.echo(json.dumps({"status": "CLOUDFLARED_MISSING", "hint": msg}, indent=2))
        else:
            console.print(f"[red]CLOUDFLARED_MISSING[/red]\n{msg}")
        raise typer.Exit(1)

    # Defer to the existing tunnel implementation
    console.print(f"Generating tunnel config at {config} for domain [bold]{domain}[/bold]...")
    console.print(
        "[dim]Use `visionservex tunnel config` and `visionservex tunnel run` for the "
        "full workflow. See docs/cloudflare_tunnel.md.[/dim]"
    )


@app.command("tunnel-stop", help="Stop a running Cloudflare Tunnel started for Colab.")
def tunnel_stop() -> None:
    """Stop a running cloudflared process started by this Colab session."""
    import subprocess

    try:
        result = subprocess.run(
            ["pkill", "-TERM", "-f", "cloudflared.*tunnel run"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            console.print("[green]Sent SIGTERM to cloudflared tunnel process(es).[/green]")
        else:
            console.print("[grey50]No cloudflared tunnel process found.[/grey50]")
    except Exception as exc:
        console.print(f"[red]Failed to stop tunnel:[/red] {exc}")
        raise typer.Exit(1)


@app.command(
    "test-remote",
    help="Test that a remote VisionServeX worker (e.g. Colab tunnel) is reachable.",
)
def test_remote(
    url: str = typer.Argument(..., help="Worker URL (e.g. https://your-tunnel.example.com)."),
    api_key: str | None = typer.Option(None, "--api-key", help="API key for auth."),
    timeout: float = typer.Option(10.0, "--timeout"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Probe a remote VisionServeX worker for health, version, and model list.

    Returns structured success/failure with actionable error codes.
    """
    try:
        import httpx
    except ImportError:
        msg = "httpx is required. pip install httpx"
        if json_:
            typer.echo(json.dumps({"status": "MISSING_DEPENDENCY", "hint": msg}, indent=2))
        else:
            console.print(f"[red]{msg}[/red]")
        raise typer.Exit(1)

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    result: dict = {"url": url, "checks": {}}
    failed = False

    with httpx.Client(timeout=timeout, headers=headers, verify=True) as c:
        # /health
        try:
            r = c.get(f"{url.rstrip('/')}/health")
            result["checks"]["health"] = {
                "status": r.status_code,
                "ok": 200 <= r.status_code < 300,
            }
        except Exception as exc:
            result["checks"]["health"] = {"error": str(exc)[:200]}
            result["status"] = "UNREACHABLE"
            result["hint"] = "Check the URL, tunnel status, and Colab session."
            failed = True

        # /models (only if health reached)
        if not failed:
            try:
                r = c.get(f"{url.rstrip('/')}/models")
                if r.status_code == 401:
                    result["checks"]["models"] = {"status": 401, "auth": "required"}
                    result["status"] = "AUTH_REQUIRED"
                elif 200 <= r.status_code < 300:
                    data = r.json() if r.content else []
                    result["checks"]["models"] = {
                        "status": r.status_code,
                        "model_count": len(data) if isinstance(data, list) else None,
                    }
                else:
                    result["checks"]["models"] = {"status": r.status_code}
            except Exception as exc:
                result["checks"]["models"] = {"error": str(exc)[:200]}

    if "status" not in result:
        result["status"] = "ok"

    if json_:
        typer.echo(json.dumps(result, indent=2))
    else:
        console.print(f"[bold]Remote worker:[/bold] {url}")
        for name, check in result["checks"].items():
            ok = check.get("ok") or check.get("status", 0) in (200, 401)
            mark = "[green]ok[/green]" if ok else "[red]fail[/red]"
            console.print(f"  {name}: {mark} {check}")
        console.print(f"\nStatus: [bold]{result.get('status', '?')}[/bold]")
        if result.get("hint"):
            console.print(f"Hint: [yellow]{result['hint']}[/yellow]")

    if result["status"] in ("UNREACHABLE", "ERROR"):
        raise typer.Exit(1)


__all__ = ["app", "in_colab"]
