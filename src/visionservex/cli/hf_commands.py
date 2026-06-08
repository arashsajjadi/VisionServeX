# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""``visionservex hf`` — Hugging Face connection (BYOT) CLI.

Every command is token-safe: the token is never printed, logged, or persisted by
VisionServeX. ``connect`` uses your own token (from an env var or a private file)
and stores it only in the standard Hugging Face cache, exactly like
``huggingface-cli login`` — never in this repo, notebooks, reports, or CI.
"""

from __future__ import annotations

import json
import os

import typer
from rich.console import Console
from rich.table import Table

from visionservex import hf_auth as H

app = typer.Typer(
    help="Connect your own Hugging Face account (BYOT) for gated/license-required models.",
    no_args_is_help=True,
)
console = Console()


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


@app.command("status")
def status_cmd(
    json_: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Show whether a Hugging Face login exists (never shows the token)."""
    source = H.hf_token_source()
    logged_in = H.hf_is_logged_in()
    payload = {
        "logged_in": logged_in,
        "token_source": source,
        "token_redacted": H.hf_get_token(redact=True),
        "hf_hub_installed": _hf_hub_installed(),
    }
    if logged_in:
        who = H.hf_whoami()
        payload.update(
            {
                "name": who.get("name"),
                "type": who.get("type"),
                "token_display_name": who.get("token_display_name"),
                "token_role": who.get("token_role"),
                "orgs": who.get("orgs"),
                "whoami_error": who.get("error"),
            }
        )
    if json_:
        _print_json(payload)
        return
    table = Table(title="Hugging Face connection", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Logged in", "[green]yes[/green]" if logged_in else "[yellow]no[/yellow]")
    table.add_row("Token source", str(source or "—"))
    table.add_row("Token (redacted)", payload["token_redacted"] or "—")
    if logged_in:
        table.add_row("User", str(payload.get("name") or "—"))
        table.add_row("Token name", str(payload.get("token_display_name") or "—"))
        table.add_row("Token role", str(payload.get("token_role") or "—"))
        if payload.get("orgs"):
            table.add_row("Orgs", ", ".join(payload["orgs"]))
    console.print(table)
    if not logged_in:
        console.print(
            "\n[dim]Not connected. Run[/dim] [cyan]visionservex hf connect[/cyan] "
            "[dim]or[/dim] [cyan]huggingface-cli login[/cyan]."
        )


@app.command("whoami")
def whoami_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    """Show your Hub username / org (token is always redacted)."""
    who = H.hf_whoami()
    if json_:
        _print_json(who)
        return
    if not who.get("logged_in"):
        console.print("[yellow]Not logged in.[/yellow] Run: [cyan]visionservex hf connect[/cyan]")
        raise typer.Exit(1)
    if who.get("error"):
        console.print(f"[red]whoami failed:[/red] {who['error']}")
        raise typer.Exit(1)
    console.print(
        f"[green]✓[/green] [bold]{who.get('name')}[/bold] ({who.get('type')}) — "
        f"token '{who.get('token_display_name') or '?'}' "
        f"[dim]({who.get('token_redacted')})[/dim]"
    )
    if who.get("orgs"):
        console.print(f"  orgs: {', '.join(who['orgs'])}")


@app.command("connect")
def connect_cmd(
    token_env: str | None = typer.Option(
        None, "--token-env", help="Name of an env var that holds your HF token (e.g. HF_TOKEN)."
    ),
    token_file: str | None = typer.Option(
        None, "--token-file", help="Path to a private file containing your HF token."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Connect your own token. Reads from --token-env / --token-file (never printed),
    or reports the existing login. Stored only in the standard HF cache."""
    token: str | None = None
    src = ""
    if token_env:
        token = os.environ.get(token_env)
        src = f"env:{token_env}"
        if not token:
            _fail(json_, f"env var {token_env} is not set or empty", code="NO_TOKEN_IN_ENV")
            return
    elif token_file:
        try:
            with open(token_file, encoding="utf-8", errors="ignore") as fh:
                token = fh.read().strip()
        except OSError as exc:
            _fail(json_, f"cannot read token file: {exc}", code="TOKEN_FILE_UNREADABLE")
            return
        src = f"file:{os.path.basename(token_file)}"
        if not (token.startswith("hf_") and len(token) >= 20):
            _fail(json_, "file does not contain a valid hf_ token", code="TOKEN_FILE_INVALID")
            return

    if token:
        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
        except Exception as exc:
            _fail(json_, f"login failed: {type(exc).__name__}: {exc}", code="LOGIN_FAILED")
            return

    # Re-detect after (possibly) logging in.
    if not H.hf_is_logged_in():
        payload = {
            "connected": False,
            "instructions": [
                "No token detected. Provide one of:",
                "  visionservex hf connect --token-env HF_TOKEN",
                "  visionservex hf connect --token-file /path/to/token.txt",
                "  huggingface-cli login",
                "Create a token (scope: read) at https://huggingface.co/settings/tokens",
            ],
        }
        if json_:
            _print_json(payload)
        else:
            console.print("[yellow]Not connected.[/yellow]")
            for line in payload["instructions"]:
                console.print(f"  {line}")
        raise typer.Exit(1)

    who = H.hf_whoami()
    payload = {
        "connected": True,
        "source": src or H.hf_token_source(),
        "token_redacted": who.get("token_redacted"),
        "name": who.get("name"),
        "token_display_name": who.get("token_display_name"),
    }
    if json_:
        _print_json(payload)
    else:
        console.print(
            f"[green]✓ connected[/green] as [bold]{who.get('name')}[/bold] "
            f"[dim]({who.get('token_redacted')})[/dim]"
        )
        console.print("[dim]Token stored only in the standard HF cache — never in this repo.[/dim]")


@app.command("logout")
def logout_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    """Remove the locally cached HF login (env vars / files stay yours to clear)."""
    res = H.hf_logout_local()
    if json_:
        _print_json(res)
        return
    if res.get("logged_out_cache"):
        console.print("[green]✓[/green] cleared cached Hugging Face login")
    if res.get("note"):
        console.print(f"[yellow]note:[/yellow] {res['note']}")


@app.command("check-model")
def check_model_cmd(
    model: str = typer.Argument(
        ..., help="A VisionServeX model id OR a Hub repo (e.g. facebook/sam3)."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Check whether your token grants access to a (gated) model — no download."""
    # Direct Hub repo id (contains a slash) -> raw access probe.
    payload = _probe_repo(model) if "/" in model else H.hf_model_access_status(model)
    if json_:
        _print_json(payload)
        return
    state = payload.get("state", "?")
    color = {
        "access_granted": "green",
        "external_api_only": "cyan",
    }.get(state, "yellow")
    console.print(f"[bold]{model}[/bold] → [{color}]{state}[/{color}]")
    for key in ("final_policy", "hf_repo", "gated"):
        if payload.get(key) is not None and payload.get(key) != "":
            console.print(f"  {key}: {payload[key]}")
    if payload.get("warning"):
        console.print(f"  [yellow]{payload['warning']}[/yellow]")
    if payload.get("next_command"):
        console.print(f"  [dim]next:[/dim] [cyan]{payload['next_command']}[/cyan]")
    acc = payload.get("acceptance")
    if isinstance(acc, dict) and acc.get("instructions"):
        console.print("  [dim]how to accept the license:[/dim]")
        for step in acc["instructions"]:
            console.print(f"    {step}")


def _probe_repo(repo: str) -> dict:
    """Raw metadata-only access probe for an arbitrary Hub repo id."""
    out: dict[str, object] = {
        "model_id": repo,
        "hf_repo": repo,
        "token_present": H.hf_is_logged_in(),
        "token_source": H.hf_token_source(),
    }
    tok = H.hf_get_token()
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )

        api = HfApi(token=tok)
        try:
            # auth_check tests download authorization (not just visibility).
            api.auth_check(repo)
            out["state"] = "access_granted"
            try:
                out["gated"] = getattr(api.model_info(repo, files_metadata=False), "gated", None)
            except Exception:
                out["gated"] = None
        except GatedRepoError:
            out["state"] = "auth_required_license_pending"
            out["next_command"] = f"Accept the license at https://huggingface.co/{repo}"
        except RepositoryNotFoundError:
            out["state"] = "not_found_or_no_access"
        except HfHubHTTPError as exc:
            code = getattr(getattr(exc, "response", None), "status_code", "?")
            out["state"] = f"http_{code}"
    except ImportError:
        out["state"] = "hf_hub_not_installed"
        out["next_command"] = "pip install 'visionservex[hf]'"
    return out


def _hf_hub_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("huggingface_hub") is not None


def _fail(json_: bool, message: str, *, code: str) -> None:
    if json_:
        _print_json({"error": {"code": code, "message": message}})
    else:
        console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(1)


__all__ = ["app"]
