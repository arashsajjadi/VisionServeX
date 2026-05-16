# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""SAM3 / SAM3.1 auth-aware wrapper.

We do NOT fake SAM3 inference. The Meta SAM3 release is gated and the model
weights may require either Hugging Face login or signed access at the
facebookresearch namespace. This module exposes structured status, auth
detection, and a login-help command — and refuses to fall back to mock
output.

Reference:
- https://github.com/facebookresearch/sam3
- https://huggingface.co/facebook/sam3
- https://huggingface.co/facebook/sam3.1
- https://ai.meta.com/blog/segment-anything-model-3/
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="SAM 3 / SAM 3.1 status, auth, and gated-access workflow.", no_args_is_help=True
)
console = Console()

# Structured error codes returned by this wrapper.
SAM3_STATUS_CODES = {
    "HF_AUTH_REQUIRED": "Hugging Face token is required to access the model repo.",
    "MODEL_ACCESS_GATED": "Model is gated at the facebookresearch namespace; accept terms on the HF model page first.",
    "SAM3_REPO_REQUIRED": "Upstream facebookresearch/sam3 repo not installed locally.",
    "CHECKPOINT_REQUIRED": "Model checkpoint not cached and auto_pull is disabled.",
    "PROMPT_TYPE_UNSUPPORTED": "Requested prompt type (point/box/text/exemplar/video) is not supported by this wrapper yet.",
}


@dataclass
class SAM3Status:
    """Structured status snapshot for a SAM3 / SAM3.1 model."""

    model_id: str
    hf_repo: str
    upstream_repo: str
    has_hf_token: bool
    hf_token_redacted: str
    checkpoint_cached: bool
    sam3_repo_installed: bool
    transformers_installed: bool
    blocker: str = ""
    blocker_code: str = ""
    fix: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_SAM3_MODELS = {
    "sam3": "facebook/sam3",
    "sam3-base": "facebook/sam3",
    "sam3-large": "facebook/sam3",
    "sam3.1": "facebook/sam3.1",
    "sam3.1-small": "facebook/sam3.1",
    "sam3.1-base-plus": "facebook/sam3.1",
    "sam3.1-large": "facebook/sam3.1",
}


def _redact(token: str | None) -> str:
    """Show only first 3 and last 2 chars of an HF token. Never log full token."""
    if not token:
        return ""
    if len(token) < 8:
        return "***"
    return f"{token[:3]}***{token[-2:]}"


def collect_sam3_status(model_id: str) -> SAM3Status:
    """Build an honest SAM3Status snapshot without performing a download."""
    hf_repo = _SAM3_MODELS.get(model_id, "facebook/sam3")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""

    transformers_installed = False
    try:
        import transformers  # noqa: F401  # type: ignore

        transformers_installed = True
    except ImportError:
        pass

    sam3_repo_installed = False
    try:
        import sam3  # noqa: F401  # type: ignore

        sam3_repo_installed = True
    except ImportError:
        pass

    checkpoint_cached = False
    try:
        from visionservex.registry import default_registry
        from visionservex.runtime.downloads import is_cached

        reg = default_registry()
        try:
            entry = reg.get(model_id)
            checkpoint_cached = is_cached(entry)
        except Exception:
            pass
    except ImportError:
        pass

    status = SAM3Status(
        model_id=model_id,
        hf_repo=hf_repo,
        upstream_repo="https://github.com/facebookresearch/sam3",
        has_hf_token=bool(hf_token),
        hf_token_redacted=_redact(hf_token),
        checkpoint_cached=checkpoint_cached,
        sam3_repo_installed=sam3_repo_installed,
        transformers_installed=transformers_installed,
    )

    if not transformers_installed:
        status.blocker_code = "HF_TRANSFORMERS_REQUIRED"
        status.blocker = "transformers is not installed"
        status.fix = "pip install 'visionservex[hf]'"
    elif not hf_token:
        status.blocker_code = "HF_AUTH_REQUIRED"
        status.blocker = SAM3_STATUS_CODES["HF_AUTH_REQUIRED"]
        status.fix = (
            "Run: huggingface-cli login   "
            "(or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN env var; "
            "tokens are redacted in all VisionServeX logs)"
        )
    elif not checkpoint_cached:
        status.blocker_code = "MODEL_ACCESS_GATED"
        status.blocker = SAM3_STATUS_CODES["MODEL_ACCESS_GATED"]
        status.fix = (
            f"1) Visit https://huggingface.co/{hf_repo} and accept access terms. "
            f"2) Run: visionservex model pull {model_id}  (with HF_TOKEN set)."
        )
    else:
        status.blocker_code = "PROMPT_TYPE_UNSUPPORTED"
        status.blocker = (
            "Checkpoint is cached, but the SAM3 engine glue is not yet implemented "
            "in VisionServeX. Use the upstream facebookresearch/sam3 repo directly."
        )
        status.fix = (
            "git clone https://github.com/facebookresearch/sam3 && cd sam3 && pip install -e ."
        )

    return status


@app.command("status")
def status_cmd(
    model_id: str = typer.Option(
        "sam3.1-base-plus",
        "--model",
        help="SAM3 model id (sam3, sam3-base, sam3-large, sam3.1, sam3.1-{small,base-plus,large}).",
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Show structured status: auth token presence, cache, repo install."""
    s = collect_sam3_status(model_id)
    if json_:
        print(json.dumps(s.to_dict(), indent=2))
        return

    table = Table(title=f"SAM3 status — {s.model_id}", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("HF repo", s.hf_repo)
    table.add_row("Upstream", s.upstream_repo)
    table.add_row("transformers installed", "yes" if s.transformers_installed else "no")
    table.add_row("sam3 repo installed", "yes" if s.sam3_repo_installed else "no")
    table.add_row("HF token present", "yes" if s.has_hf_token else "no")
    table.add_row("HF token (redacted)", s.hf_token_redacted or "—")
    table.add_row("Checkpoint cached", "yes" if s.checkpoint_cached else "no")
    table.add_row("Blocker", f"[yellow]{s.blocker_code}[/yellow]" if s.blocker_code else "—")
    table.add_row("Details", s.blocker or "—")
    table.add_row("Fix", s.fix or "—")
    console.print(table)


@app.command("login-help")
def login_help() -> None:
    """Print exactly how to authenticate with Hugging Face for gated SAM3 access."""
    console.print(
        "[bold]Hugging Face login for SAM3:[/bold]\n"
        "\n"
        "  1. Create a token at: https://huggingface.co/settings/tokens  (scope: 'read').\n"
        "  2. Authenticate locally — either:\n"
        "       [cyan]huggingface-cli login[/cyan]\n"
        "     or set an env var (never commit this):\n"
        "       export HF_TOKEN=hf_...\n"
        "  3. Accept the SAM3 model's access terms on its model page:\n"
        "       https://huggingface.co/facebook/sam3\n"
        "       https://huggingface.co/facebook/sam3.1\n"
        "  4. Confirm: [cyan]visionservex sam3 status --model sam3.1-base-plus[/cyan]\n"
        "\n"
        "All VisionServeX logs redact HF tokens automatically — only the first 3 and "
        "last 2 chars are ever shown."
    )


@app.command("supported-prompts")
def supported_prompts(json_: bool = typer.Option(False, "--json")) -> None:
    """List which SAM3 prompt types are wired in VisionServeX today."""
    payload = {
        "wired": [],
        "not_wired": ["point", "box", "text", "exemplar", "video"],
        "note": (
            "VisionServeX does not include SAM3 inference glue in v1.8.0. "
            "The auth/status wrapper is provided so users know exactly what is missing. "
            "Use facebookresearch/sam3 directly for inference."
        ),
    }
    if json_:
        print(json.dumps(payload, indent=2))
        return
    console.print(
        "[bold]SAM3 prompt support in VisionServeX v1.8.0:[/bold]\n"
        f"  wired:     {payload['wired'] or '(none)'}\n"
        f"  not wired: {payload['not_wired']}\n"
        f"\n[dim]{payload['note']}[/dim]"
    )


__all__ = ["SAM3Status", "app", "collect_sam3_status"]
