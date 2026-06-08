# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Model lifecycle CLI: pull, cache, checkpoint-info, verify, remove, list-local."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Model lifecycle: pull, cache, checkpoint info, verify, register.")
console = Console()


@app.command("info", help="Show full model info from registry + cache status.")
def model_info(
    model_id: str,
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.registry import RegistryError, default_registry
    from visionservex.runtime.downloads import cached_path, is_cached

    try:
        entry = default_registry().get(model_id)
    except RegistryError as exc:
        if json_:
            typer.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1)

    cp = cached_path(entry)
    size_bytes = 0
    if cp and cp.exists():
        if cp.is_dir():
            size_bytes = sum(f.stat().st_size for f in cp.rglob("*") if f.is_file())
        else:
            size_bytes = cp.stat().st_size

    payload = entry.model_dump()
    payload["cached"] = is_cached(entry)
    payload["cache_path"] = str(cp) if cp else None
    payload["cache_size_mb"] = round(size_bytes / (1024 * 1024), 2) if size_bytes else 0
    payload["checkpoint_source"] = entry.download_type
    payload["checkpoint_trust_level"] = (
        "community_hf"
        if entry.download_type == "huggingface"
        else "package_managed"
        if entry.download_type == "package_managed"
        else "manual"
    )

    if json_:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    from rich.panel import Panel

    cat_color = {
        "accuracy_grade": "green",
        "production_recommended": "cyan",
        "demo_fast": "yellow",
        "experimental_sota": "magenta",
        "expert_sidecar": "grey50",
        "unavailable_with_reason": "red",
    }.get(entry.model_category or "", "white")

    console.print(
        Panel.fit(
            f"[bold]{entry.display_name}[/bold]\n"
            f"ID: {entry.id} | Task: {entry.task} | "
            f"Category: [{cat_color}]{entry.model_category}[/{cat_color}]",
            border_style="cyan",
        )
    )
    cached = "[green]yes[/green]" if is_cached(entry) else "[grey50]no[/grey50]"
    console.print(f"  Cached:      {cached}")
    if cp:
        console.print(f"  Cache path:  {cp}")
        if size_bytes:
            console.print(f"  Size:        {size_bytes / (1024 * 1024):.1f} MB")
    console.print(
        f"  License:     {entry.license}{'  ⚠ uncertain' if entry.license_uncertain else ''}"
    )
    console.print(f"  Status:      {entry.status} / {entry.implementation_status}")
    console.print(f"  Upstream:    {entry.upstream_url}")
    if entry.auto_download:
        console.print(f"\n  [cyan]$[/cyan] visionservex model pull {entry.id}")
    else:
        console.print(f"\n  [yellow]Manual download required.[/yellow] See: {entry.upstream_url}")


@app.command("pull", help="Download model checkpoint (license-policy enforced).")
def pull_model(
    model_id: str,
    force: bool = typer.Option(False, "--force", help="Re-download even if cached."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be downloaded without downloading."
    ),
    accept_upstream_license: bool = typer.Option(
        False,
        "--accept-upstream-license",
        help="Confirm you accepted the upstream license yourself (required for BYOT/gated models).",
    ),
    research_only: bool = typer.Option(
        False, "--research-only", help="Run a non-commercial model for research only."
    ),
    accept_noncommercial: bool = typer.Option(
        False,
        "--accept-noncommercial",
        help="Acknowledge a non-commercial license (required with --research-only).",
    ),
    hf_token_env: str | None = typer.Option(
        None, "--hf-token-env", help="Env var holding your HF token (BYOT). Token never printed."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    # ----- license-policy gate (only for models in the v3.8 policy table) -----
    from visionservex import hf_auth as _H
    from visionservex.licensing import policy as _pol
    from visionservex.registry import RegistryError, default_registry
    from visionservex.runtime.downloads import (
        DownloadError,
        ManualDownloadRequired,
        download,
        is_cached,
    )

    if hf_token_env:
        import os as _os

        if not _os.environ.get(hf_token_env):
            _die(
                f"--hf-token-env {hf_token_env} is not set", json_mode=json_, code="NO_TOKEN_IN_ENV"
            )
            return

    pol = _pol.get_policy(model_id)
    if pol is not None:
        canonical = _pol.resolve_model_id(model_id)
        fp = pol.final_policy
        if fp == "byot_license_required":
            if not accept_upstream_license:
                _die(
                    f"{canonical} is gated/BYOT. {pol.warning_text}",
                    json_mode=json_,
                    code="UPSTREAM_LICENSE_NOT_ACCEPTED",
                    exit_code=2,
                    extra={
                        "next_command": f"visionservex model pull {canonical} --accept-upstream-license",
                        "upstream_url": pol.upstream_url,
                        "acceptance": _H.hf_acceptance_instructions(canonical)["instructions"],
                    },
                )
                return
            try:
                _H.hf_require_user_accepted_license(canonical)
            except _H.HFLicenseError as exc:
                _die(
                    str(exc),
                    json_mode=json_,
                    code=exc.state.upper(),
                    exit_code=2,
                    extra={"next_command": exc.next_command},
                )
                return
            if dry_run:
                _emit_pull(
                    {
                        "model_id": canonical,
                        "hf_repo": pol.hf_repo,
                        "final_policy": fp,
                        "access": "granted",
                        "dry_run": True,
                    },
                    json_,
                )
                return
            path = _byot_snapshot(pol.hf_repo, json_)
            _emit_pull(
                {
                    "model_id": canonical,
                    "hf_repo": pol.hf_repo,
                    "path": str(path),
                    "final_policy": fp,
                    "status": "ok",
                    "warning": pol.warning_text,
                },
                json_,
            )
            return
        if fp == "noncommercial_restricted":
            if not (research_only and accept_noncommercial):
                _die(
                    f"{canonical}: {pol.warning_text}",
                    json_mode=json_,
                    code="NONCOMMERCIAL_REFUSED",
                    exit_code=2,
                    extra={
                        "next_command": (
                            f"visionservex model pull {canonical} --research-only --accept-noncommercial"
                        )
                    },
                )
                return
            # research-only path falls through to registry/HF download if available
        elif fp in (
            "enterprise_license_required",
            "legal_review_required",
            "external_api_only_terms_required",
            "not_released_or_unverifiable",
        ):
            _die(
                f"{canonical}: {pol.warning_text}",
                json_mode=json_,
                code=fp.upper(),
                exit_code=2,
                extra={"next_command": pol.exact_next_command, "upstream_url": pol.upstream_url},
            )
            return
        # commercial_safe_core falls through to the normal registry download.

    try:
        entry = default_registry().get(model_id)
    except RegistryError as exc:
        # Policy model that isn't in the runtime registry but is pullable from HF.
        if (
            pol is not None
            and pol.hf_repo
            and pol.final_policy in ("commercial_safe_core", "noncommercial_restricted")
        ):
            if dry_run:
                _emit_pull({"model_id": model_id, "hf_repo": pol.hf_repo, "dry_run": True}, json_)
                return
            path = _byot_snapshot(pol.hf_repo, json_)
            _emit_pull(
                {"model_id": model_id, "hf_repo": pol.hf_repo, "path": str(path), "status": "ok"},
                json_,
            )
            return
        _die(str(exc), json_mode=json_, code="MODEL_NOT_FOUND")
        return

    if dry_run:
        payload = {
            "model_id": model_id,
            "download_type": entry.download_type,
            "hf_repo_id": getattr(entry, "hf_repo_id", None),
            "auto_download": entry.auto_download,
            "already_cached": is_cached(entry),
            "dry_run": True,
        }
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(f"[dim]dry-run:[/dim] would download {model_id}")
            console.print(f"  Source: {entry.download_type}")
            if getattr(entry, "hf_repo_id", None):
                console.print(f"  HF repo: {entry.hf_repo_id}")
            console.print(f"  Already cached: {'yes' if is_cached(entry) else 'no'}")
        return

    try:
        path = download(entry, force=force)
        payload = {"model_id": model_id, "path": str(path), "status": "ok"}
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(f"[green]✓[/green] {model_id} → {path}")
    except ManualDownloadRequired as exc:
        _die(str(exc), json_mode=json_, code="MANUAL_DOWNLOAD_REQUIRED", exit_code=2)
    except DownloadError as exc:
        _die(str(exc), json_mode=json_, code="DOWNLOAD_FAILED")


@app.command("checkpoint-info", help="Show checkpoint provenance and trust metadata.")
def checkpoint_info(
    model_id: str,
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.registry import RegistryError, default_registry
    from visionservex.runtime.downloads import cached_path, is_cached

    try:
        entry = default_registry().get(model_id)
    except RegistryError as exc:
        _die(str(exc), json_mode=json_, code="MODEL_NOT_FOUND")
        return

    cp = cached_path(entry)
    payload = {
        "model_id": model_id,
        "source": entry.download_type,
        "hf_repo_id": getattr(entry, "hf_repo_id", None),
        "upstream_url": entry.upstream_url,
        "license": entry.license,
        "license_uncertain": entry.license_uncertain or False,
        "implementation_status": entry.implementation_status,
        "checkpoint_trust_level": (
            "community_hf"
            if entry.download_type == "huggingface"
            else "package_managed"
            if entry.download_type == "package_managed"
            else "manual"
            if entry.download_type == "manual"
            else "synthetic"
        ),
        "cached": is_cached(entry),
        "cache_path": str(cp) if cp else None,
        "official_ap_claim": "see model-card for upstream benchmark claims",
        "verified_by_visionservex": "latency_tested_only — use benchmark-competitiveness --dataset for AP",
    }

    if json_:
        typer.echo(json.dumps(payload, indent=2))
        return

    console.print(f"[bold]Checkpoint info:[/bold] {model_id}")
    console.print(f"  Source:          {payload['source']}")
    if payload["hf_repo_id"]:
        console.print(f"  HF repo:         {payload['hf_repo_id']}")
    console.print(
        f"  License:         {payload['license']}{'  ⚠ uncertain' if payload['license_uncertain'] else ''}"
    )
    console.print(f"  Trust level:     {payload['checkpoint_trust_level']}")
    console.print(f"  Cached:          {'yes' if payload['cached'] else 'no'}")
    if payload["cache_path"]:
        console.print(f"  Cache path:      {payload['cache_path']}")
    console.print(f"\n  [dim]{payload['verified_by_visionservex']}[/dim]")


@app.command("cache", help="Show cache info for a model.")
def model_cache(
    model_id: str | None = typer.Argument(None, help="Model ID (omit for all)."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.registry import default_registry
    from visionservex.runtime.downloads import cache_listing, cached_path, is_cached

    if model_id:
        try:
            entry = default_registry().get(model_id)
        except Exception as exc:
            _die(str(exc), json_mode=json_, code="MODEL_NOT_FOUND")
            return
        cp = cached_path(entry)
        size = 0
        if cp and cp.exists():
            if cp.is_dir():
                size = sum(f.stat().st_size for f in cp.rglob("*") if f.is_file())
            else:
                size = cp.stat().st_size
        payload = {
            "model_id": model_id,
            "cached": is_cached(entry),
            "path": str(cp) if cp else None,
            "size_mb": round(size / (1024 * 1024), 2),
        }
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            cached = "[green]yes[/green]" if payload["cached"] else "[grey50]no[/grey50]"
            console.print(
                f"{model_id}: cached={cached} path={payload['path']} size={payload['size_mb']} MB"
            )
        return

    items = cache_listing()
    if json_:
        typer.echo(json.dumps(items, indent=2))
        return
    if not items:
        console.print("No models cached.")
        return
    table = Table(title="Cached models")
    table.add_column("ID")
    table.add_column("Size MiB")
    table.add_column("Path")
    for item in items:
        table.add_row(item["model_id"], f"{item['size_bytes'] / (1024 * 1024):.1f}", item["path"])
    console.print(table)


@app.command("verify", help="Verify cached model files.")
def verify_model(
    model_id: str | None = typer.Argument(None),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.runtime.downloads import cache_verify

    report = cache_verify(model_id)
    if json_:
        typer.echo(json.dumps(report, indent=2))
        return
    if not report:
        console.print("nothing cached to verify")
        return
    for r in report:
        status = "[green]ok[/green]" if r["ok"] else "[red]bad[/red]"
        console.print(f"  {r['model_id']}: {status} — {r['reason']}")


@app.command("clear-cache", help="Delete cached files for a model.")
def clear_cache_cmd(
    model_id: str,
    yes: bool = typer.Option(False, "--yes"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    if not yes and not typer.confirm(f"Delete cached files for {model_id}?"):
        raise typer.Exit(1)
    from visionservex.runtime.downloads import cache_clean

    freed = cache_clean(model_id)
    payload = {"model_id": model_id, "bytes_freed": freed}
    if json_:
        typer.echo(json.dumps(payload, indent=2))
    else:
        console.print(f"[green]freed {freed / (1024 * 1024):.1f} MiB[/green] from {model_id}")


@app.command("list-local", help="List all locally cached models.")
def list_local(json_: bool = typer.Option(False, "--json")) -> None:
    from visionservex.runtime.downloads import cache_listing

    items = cache_listing()
    if json_:
        typer.echo(json.dumps(items, indent=2))
        return
    if not items:
        console.print("[grey50]No models cached.[/grey50]")
        return
    table = Table(title=f"Locally cached models ({len(items)})")
    table.add_column("Model ID")
    table.add_column("Size")
    table.add_column("Path")
    for item in items:
        size = f"{item['size_bytes'] / (1024 * 1024):.1f} MiB"
        table.add_row(item["model_id"], size, item["path"][:60])
    console.print(table)


def _die(
    message: str,
    *,
    json_mode: bool,
    code: str = "ERROR",
    exit_code: int = 1,
    extra: dict | None = None,
) -> None:
    if json_mode:
        payload = {"error": {"code": code, "message": message}}
        if extra:
            payload["error"].update(extra)
        typer.echo(json.dumps(payload, indent=2, default=str), err=True)
    else:
        console.print(f"[red]error[{code}]:[/red] {message}")
        if extra:
            if extra.get("next_command"):
                console.print(f"  [dim]next:[/dim] [cyan]{extra['next_command']}[/cyan]")
            if extra.get("upstream_url"):
                console.print(f"  [dim]upstream:[/dim] {extra['upstream_url']}")
            for step in extra.get("acceptance", []) or []:
                console.print(f"    {step}")
    raise typer.Exit(exit_code)


def _emit_pull(payload: dict, json_mode: bool) -> None:
    if json_mode:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    if payload.get("dry_run"):
        console.print(
            f"[dim]dry-run:[/dim] would pull {payload.get('model_id')} "
            f"from {payload.get('hf_repo') or 'official source'}"
        )
        return
    console.print(f"[green]✓[/green] {payload.get('model_id')} → {payload.get('path')}")
    if payload.get("warning"):
        console.print(f"  [yellow]{payload['warning']}[/yellow]")


def _byot_snapshot(repo: str, json_mode: bool):
    """Download a (gated) HF repo to the standard HF cache using the user's token.

    Weights land only in the user's Hugging Face cache — never in this repo. The
    token is read internally and never printed.
    """
    from visionservex import hf_auth as _H

    if not repo:
        _die(
            "no Hugging Face repo on record for this model", json_mode=json_mode, code="NO_HF_REPO"
        )
        return None
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _die(
            "huggingface_hub is required",
            json_mode=json_mode,
            code="HF_HUB_REQUIRED",
            extra={"next_command": "pip install 'visionservex[hf]'"},
        )
        return None
    try:
        return snapshot_download(repo_id=repo, token=_H.hf_get_token())
    except Exception as exc:
        _die(
            f"download failed: {type(exc).__name__}: {exc}",
            json_mode=json_mode,
            code="DOWNLOAD_FAILED",
        )
        return None


@app.command("license", help="Show the VisionServeX license policy for a model.")
def model_license(
    model_id: str,
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.licensing import policy as _pol

    pol = _pol.get_policy(model_id)
    if pol is None:
        _die(
            f"'{model_id}' is not in the license policy table",
            json_mode=json_,
            code="UNKNOWN_MODEL",
        )
        return
    row = pol.as_row()
    if json_:
        typer.echo(json.dumps(row, indent=2, default=str))
        return
    from rich.panel import Panel

    color = {
        "commercial_safe_core": "green",
        "byot_license_required": "cyan",
        "external_api_only_terms_required": "blue",
    }.get(pol.final_policy, "yellow")
    console.print(
        Panel.fit(
            f"[bold]{pol.model_id}[/bold]  ({pol.family})\n"
            f"final policy: [{color}]{pol.final_policy}[/{color}]",
            border_style=color,
        )
    )
    console.print(f"  code license:     {pol.code_license}")
    console.print(f"  weights license:  {pol.weights_license}")
    console.print(f"  dataset risk:     {pol.dataset_risk}")
    console.print(f"  gated:            {pol.gated}   token required: {pol.local_token_required}")
    console.print(
        f"  default_safe:     {pol.default_safe}   commercial_safe: {pol.commercial_safe}   production: {pol.production_allowed}"
    )
    console.print(
        f"  can auto-download:{pol.can_auto_download}   can ship weights: {pol.can_ship_weights}"
    )
    if pol.hf_repo:
        console.print(f"  hf repo:          {pol.hf_repo}")
    if pol.upstream_url:
        console.print(f"  upstream:         {pol.upstream_url}")
    console.print(f"\n  [yellow]{pol.warning_text}[/yellow]")
    console.print(f"  [dim]next:[/dim] [cyan]{pol.exact_next_command}[/cyan]")


@app.command("status", help="Runtime status for a model (policy + access + cache).")
def model_status(
    model_id: str,
    explain: bool = typer.Option(False, "--explain", help="Include full policy + access detail."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex import hf_auth as _H
    from visionservex.licensing import policy as _pol
    from visionservex.registry import RegistryError, default_registry
    from visionservex.runtime.downloads import is_cached

    canonical = _pol.resolve_model_id(model_id)
    pol = _pol.get_policy(model_id)
    cached = None
    try:
        entry = default_registry().get(canonical)
        cached = is_cached(entry)
    except RegistryError:
        pass
    payload: dict = {
        "model_id": canonical,
        "final_policy": pol.final_policy if pol else "not_in_policy_table",
        "cached": cached,
    }
    if pol is not None:
        payload["access"] = _H.hf_model_access_status(canonical)
        payload["state"] = payload["access"].get("state")
        if explain:
            payload["policy"] = pol.as_row()
            payload["acceptance"] = _H.hf_acceptance_instructions(canonical)
    if json_:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    console.print(f"[bold]{canonical}[/bold]")
    console.print(f"  final policy: {payload['final_policy']}")
    if payload.get("state"):
        console.print(f"  runtime state: {payload['state']}")
    console.print(f"  cached: {cached}")
    if explain and pol is not None:
        console.print(f"  [yellow]{pol.warning_text}[/yellow]")
        console.print(
            f"  [dim]next:[/dim] [cyan]{payload['access'].get('next_command', pol.exact_next_command)}[/cyan]"
        )


@app.command("doctor", help="Diagnose why a model can or cannot run, with exact fixes.")
def model_doctor(
    model_id: str,
    json_: bool = typer.Option(False, "--json"),
) -> None:
    import importlib.util

    from visionservex import hf_auth as _H
    from visionservex.licensing import policy as _pol

    canonical = _pol.resolve_model_id(model_id)
    pol = _pol.get_policy(model_id)
    checks: list[dict] = []

    def chk(name, ok, detail="", fix=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail, "fix": fix})

    chk(
        "in_license_policy",
        pol is not None,
        detail=pol.final_policy if pol else "model not in policy table",
        fix="" if pol else "visionservex model license <id> for known models",
    )
    chk(
        "huggingface_hub_installed",
        importlib.util.find_spec("huggingface_hub") is not None,
        fix="pip install 'visionservex[hf]'",
    )
    chk(
        "transformers_installed",
        importlib.util.find_spec("transformers") is not None,
        fix="pip install 'visionservex[hf]'",
    )
    chk(
        "hf_logged_in",
        _H.hf_is_logged_in(),
        detail=f"source={_H.hf_token_source()}",
        fix="visionservex hf connect",
    )
    if pol is not None and pol.gated:
        acc = _H.hf_model_access_status(canonical)
        chk(
            "gated_access_granted",
            acc.get("state") == "access_granted",
            detail=acc.get("state", ""),
            fix=acc.get("next_command", f"accept license at {pol.upstream_url}"),
        )
    overall = "ready" if all(c["ok"] for c in checks) else "blocked"
    payload = {
        "model_id": canonical,
        "final_policy": pol.final_policy if pol else None,
        "overall": overall,
        "checks": checks,
    }
    if json_:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    console.print(
        f"[bold]doctor:[/bold] {canonical} → "
        f"{'[green]ready[/green]' if overall == 'ready' else '[yellow]blocked[/yellow]'}"
    )
    for c in checks:
        mark = "[green]✓[/green]" if c["ok"] else "[red]✗[/red]"
        line = f"  {mark} {c['check']}"
        if c["detail"]:
            line += f" [dim]({c['detail']})[/dim]"
        console.print(line)
        if not c["ok"] and c["fix"]:
            console.print(f"      [dim]fix:[/dim] [cyan]{c['fix']}[/cyan]")


__all__ = ["app"]
