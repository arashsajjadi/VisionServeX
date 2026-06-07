# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""VisionServeX v3.4 — DINO family CLI.

Covers the full DINO family in a single typer app:

  dinov2-*             — runnable, Apache-2.0, no auth
  dinov3-*             — HF-gated, custom Meta license, auth required
  grounding-dino-swin-t/b  — runnable, Apache-2.0, no auth
  grounding-dino-original-* — runnable, Apache-2.0, no auth
  grounding-dino-1.5/1.6   — API-gated, DEEPDATASPACE_API_KEY required
  dino-x-api           — API-only, proprietary

Commands:
  list    — list all DINO-family models with status
  status  — JSON status card per model_id
  embed   — DINOv2 embedding → .npy
  knn     — kNN similarity search over a gallery directory
  detect  — GroundingDINO text-prompted detection → .json
  api     — API-only models (key-guarded, key never logged in full)
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="dino",
    help="DINO family: DINOv2/v3 embeddings, GroundingDINO, DINO-X (list/status/embed/knn/detect/api).",
    no_args_is_help=True,
)
console = Console()

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_DINOV2_MODELS = {
    "dinov2-small": {
        "family": "dinov2",
        "task": "embed",
        "license": "Apache-2.0",
        "runnable": True,
        "auth_required": False,
        "api_required": False,
        "hf_repo": "facebook/dinov2-small",
        "dim": 384,
        "description": "DINOv2 ViT-S/14 (384-dim embeddings)",
    },
    "dinov2-base": {
        "family": "dinov2",
        "task": "embed",
        "license": "Apache-2.0",
        "runnable": True,
        "auth_required": False,
        "api_required": False,
        "hf_repo": "facebook/dinov2-base",
        "dim": 768,
        "description": "DINOv2 ViT-B/14 (768-dim embeddings)",
    },
    "dinov2-large": {
        "family": "dinov2",
        "task": "embed",
        "license": "Apache-2.0",
        "runnable": True,
        "auth_required": False,
        "api_required": False,
        "hf_repo": "facebook/dinov2-large",
        "dim": 1024,
        "description": "DINOv2 ViT-L/14 (1024-dim embeddings)",
    },
    "dinov2-giant": {
        "family": "dinov2",
        "task": "embed",
        "license": "Apache-2.0",
        "runnable": True,
        "auth_required": False,
        "api_required": False,
        "hf_repo": "facebook/dinov2-giant",
        "dim": 1536,
        "description": "DINOv2 ViT-G/14 (1536-dim embeddings)",
    },
}

_DINOV3_MODELS = {
    "dinov3-vits16": {
        "family": "dinov3",
        "task": "embed",
        "license": "DINOv3 Meta custom license (HF-gated)",
        "runnable": False,
        "auth_required": True,
        "api_required": False,
        "code": "HF_GATED_LICENSE_REQUIRED",
        "hf_repo": "facebook/dinov3-vits16",
        "description": "DINOv3 ViT-S/16 — HF-gated, custom Meta license",
    },
    "dinov3-vitb16": {
        "family": "dinov3",
        "task": "embed",
        "license": "DINOv3 Meta custom license (HF-gated)",
        "runnable": False,
        "auth_required": True,
        "api_required": False,
        "code": "HF_GATED_LICENSE_REQUIRED",
        "hf_repo": "facebook/dinov3-vitb16",
        "description": "DINOv3 ViT-B/16 — HF-gated, custom Meta license",
    },
    "dinov3-vitl16": {
        "family": "dinov3",
        "task": "embed",
        "license": "DINOv3 Meta custom license (HF-gated)",
        "runnable": False,
        "auth_required": True,
        "api_required": False,
        "code": "HF_GATED_LICENSE_REQUIRED",
        "hf_repo": "facebook/dinov3-vitl16",
        "description": "DINOv3 ViT-L/16 — HF-gated, custom Meta license",
    },
    "dinov3-vit7b16": {
        "family": "dinov3",
        "task": "embed",
        "license": "DINOv3 Meta custom license (HF-gated)",
        "runnable": False,
        "auth_required": True,
        "api_required": False,
        "code": "HF_GATED_LICENSE_REQUIRED",
        "hf_repo": "facebook/dinov3-vit7b16",
        "description": "DINOv3 ViT-7B/16 — HF-gated, custom Meta license",
    },
}

_GROUNDING_DINO_MODELS = {
    "grounding-dino-swin-t": {
        "family": "grounding-dino",
        "task": "open_vocab_detect",
        "license": "Apache-2.0",
        "runnable": True,
        "auth_required": False,
        "api_required": False,
        "hf_repo": "IDEA-Research/grounding-dino-tiny",
        "description": "GroundingDINO Swin-T (tiny) — text-prompted detection",
    },
    "grounding-dino-swin-b": {
        "family": "grounding-dino",
        "task": "open_vocab_detect",
        "license": "Apache-2.0",
        "runnable": True,
        "auth_required": False,
        "api_required": False,
        "hf_repo": "IDEA-Research/grounding-dino-base",
        "description": "GroundingDINO Swin-B (base) — text-prompted detection",
    },
    "grounding-dino-original": {
        "family": "grounding-dino",
        "task": "open_vocab_detect",
        "license": "Apache-2.0",
        "runnable": True,
        "auth_required": False,
        "api_required": False,
        "hf_repo": "IDEA-Research/grounding-dino-tiny",
        "description": "GroundingDINO original (Swin-T) — text-prompted detection",
    },
    "grounding-dino-original-swin-t": {
        "family": "grounding-dino",
        "task": "open_vocab_detect",
        "license": "Apache-2.0",
        "runnable": True,
        "auth_required": False,
        "api_required": False,
        "hf_repo": "IDEA-Research/grounding-dino-tiny",
        "description": "GroundingDINO original Swin-T — text-prompted detection",
    },
    "grounding-dino-original-swin-b": {
        "family": "grounding-dino",
        "task": "open_vocab_detect",
        "license": "Apache-2.0",
        "runnable": True,
        "auth_required": False,
        "api_required": False,
        "hf_repo": "IDEA-Research/grounding-dino-base",
        "description": "GroundingDINO original Swin-B — text-prompted detection",
    },
    "grounding-dino-1.5": {
        "family": "grounding-dino",
        "task": "open_vocab_detect",
        "license": "DeepDataSpace proprietary (API-gated)",
        "runnable": False,
        "auth_required": True,
        "api_required": True,
        "env_var": "DEEPDATASPACE_API_KEY",
        "description": "GroundingDINO 1.5 — API-gated, requires DEEPDATASPACE_API_KEY",
    },
    "grounding-dino-1.6": {
        "family": "grounding-dino",
        "task": "open_vocab_detect",
        "license": "DeepDataSpace proprietary (API-gated)",
        "runnable": False,
        "auth_required": True,
        "api_required": True,
        "env_var": "DEEPDATASPACE_API_KEY",
        "description": "GroundingDINO 1.6 — API-gated, requires DEEPDATASPACE_API_KEY",
    },
}

_DINO_X_MODELS = {
    "dino-x-api": {
        "family": "dino-x",
        "task": "open_vocab_detect",
        "license": "proprietary/closed (API-only)",
        "runnable": False,
        "auth_required": True,
        "api_required": True,
        "env_var": "DEEPDATASPACE_API_KEY",
        "description": "DINO-X — API-only, proprietary weights, requires DEEPDATASPACE_API_KEY",
    },
}

_ALL_MODELS: dict[str, dict] = {
    **_DINOV2_MODELS,
    **_DINOV3_MODELS,
    **_GROUNDING_DINO_MODELS,
    **_DINO_X_MODELS,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _redact_key(key: str) -> str:
    """Show first 3 + *** + last 2 chars; never log the full key."""
    if not key:
        return "(empty)"
    if len(key) <= 6:
        return key[:1] + "***"
    return key[:3] + "***" + key[-2:]


def _blocker_for(meta: dict) -> str:
    """Return a human-readable blocker string or empty string when runnable."""
    if meta.get("api_required"):
        env = meta.get("env_var", "API_KEY")
        return f"API key required — set ${env}"
    if meta.get("auth_required"):
        code = meta.get("code", "HF_GATED_LICENSE_REQUIRED")
        return f"{code} — set $HF_TOKEN and request HF gated access"
    if not meta.get("runnable", False):
        return "NOT_RUNNABLE — unknown blocker"
    return ""


def _fix_for(meta: dict) -> str:
    """Return the concrete fix command."""
    if meta.get("api_required"):
        env = meta.get("env_var", "API_KEY")
        return f"export {env}=<your-key> && visionservex dino api {meta.get('model_id', '')} image.jpg --text '...'"
    if meta.get("auth_required"):
        return "export HF_TOKEN=<token> && request access at https://huggingface.co/" + meta.get("hf_repo", "")
    return ""


def _status_card(model_id: str, meta: dict) -> dict:
    blocker = _blocker_for(meta)
    return {
        "model_id": model_id,
        "family": meta["family"],
        "task": meta["task"],
        "license": meta["license"],
        "description": meta.get("description", ""),
        "runnable": meta.get("runnable", False),
        "auth_required": meta.get("auth_required", False),
        "api_required": meta.get("api_required", False),
        "blocker": blocker,
        "fix": _fix_for({**meta, "model_id": model_id}),
        "code": meta.get("code", "") or ("API_REQUIRED" if meta.get("api_required") else ""),
        "env_var": meta.get("env_var", ""),
        "hf_repo": meta.get("hf_repo", ""),
    }


def _emit(payload: object, *, json_out: bool, explain: bool = False) -> None:
    """Print JSON or a human-readable summary."""
    if json_out:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    if isinstance(payload, list):
        for item in payload:
            _print_card(item, explain=explain)
    elif isinstance(payload, dict):
        _print_card(payload, explain=explain)
    else:
        typer.echo(str(payload))


def _print_card(card: dict, *, explain: bool = False) -> None:
    mid = card.get("model_id", "?")
    family = card.get("family", "?")
    runnable = card.get("runnable", False)
    status_color = "green" if runnable else "yellow"
    status_label = "runnable" if runnable else "blocked"
    console.print(f"[bold]{mid}[/bold]  [{status_color}]{status_label}[/{status_color}]  family={family}")
    if explain:
        console.print(f"  license    : {card.get('license', '?')}")
        console.print(f"  task       : {card.get('task', '?')}")
        console.print(f"  auth_req   : {card.get('auth_required', False)}")
        console.print(f"  api_req    : {card.get('api_required', False)}")
        if card.get("blocker"):
            console.print(f"  [yellow]blocker[/yellow]  : {card['blocker']}")
        if card.get("fix"):
            console.print(f"  fix        : {card['fix']}")
        if card.get("description"):
            console.print(f"  description: {card['description']}")


def _require_model(model_id: str) -> dict:
    meta = _ALL_MODELS.get(model_id)
    if meta is None:
        payload = {
            "status": "error",
            "code": "UNKNOWN_MODEL",
            "model_id": model_id,
            "message": f"Unknown DINO model {model_id!r}.",
            "available": sorted(_ALL_MODELS),
        }
        typer.echo(json.dumps(payload, indent=2))
        raise typer.Exit(2)
    return meta


def _probe(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command("list")
def list_cmd(
    json_: bool = typer.Option(False, "--json", help="Emit JSON array."),
    explain: bool = typer.Option(False, "--explain", help="Show license/auth/blocker detail."),
) -> None:
    """List all DINO-family models with status."""
    rows = [_status_card(mid, meta) for mid, meta in _ALL_MODELS.items()]
    _emit(rows, json_out=json_, explain=explain)


@app.command("status")
def status_cmd(
    model_id: str = typer.Argument(..., help="Model ID (e.g. dinov2-base, grounding-dino-swin-t)."),
    json_: bool = typer.Option(False, "--json", help="Emit JSON."),
    explain: bool = typer.Option(False, "--explain", help="Print explain card."),
) -> None:
    """Return a JSON status card for a DINO-family model."""
    meta = _require_model(model_id)
    card = _status_card(model_id, meta)
    _emit(card, json_out=json_, explain=explain)


@app.command("embed")
def embed_cmd(
    model_id: str = typer.Argument(..., help="DINOv2 model ID (dinov2-small/base/large/giant)."),
    image: str = typer.Argument(..., help="Path to input image."),
    out: Path = typer.Option(Path("embedding.npy"), "--out", help="Output .npy file path."),
    device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda."),
    json_: bool = typer.Option(False, "--json", help="Emit JSON result."),
    explain: bool = typer.Option(False, "--explain", help="Print model explain card and exit."),
) -> None:
    """Run DINOv2 embedding on an image and save the result as a .npy file."""
    meta = _require_model(model_id)

    if explain:
        card = _status_card(model_id, meta)
        _emit(card, json_out=json_, explain=True)
        return

    if not meta.get("runnable", False):
        payload = {
            "status": "blocked",
            "code": meta.get("code") or "NOT_RUNNABLE",
            "model_id": model_id,
            "blocker": _blocker_for(meta),
            "fix": _fix_for({**meta, "model_id": model_id}),
        }
        _emit(payload, json_out=True)
        raise typer.Exit(3)

    if meta["family"] != "dinov2":
        payload = {
            "status": "error",
            "code": "WRONG_TASK",
            "model_id": model_id,
            "message": f"{model_id!r} is not a DINOv2 embedding model. "
                       f"Use dinov2-small/base/large/giant for embed.",
        }
        _emit(payload, json_out=True)
        raise typer.Exit(2)

    image_path = Path(image)
    if not image_path.exists():
        payload = {
            "status": "error",
            "code": "INPUT_NOT_FOUND",
            "image": image,
            "message": f"Image not found: {image}",
        }
        _emit(payload, json_out=True)
        raise typer.Exit(2)

    # Probe required deps before attempting inference.
    if not _probe("torch"):
        payload = {
            "status": "blocked",
            "code": "TORCH_REQUIRED",
            "model_id": model_id,
            "message": "PyTorch is required. Install with: pip install torch",
            "fix": "pip install torch torchvision",
        }
        _emit(payload, json_out=True)
        raise typer.Exit(3)

    if not _probe("transformers"):
        payload = {
            "status": "blocked",
            "code": "TRANSFORMERS_REQUIRED",
            "model_id": model_id,
            "message": "HuggingFace transformers is required.",
            "fix": "pip install 'visionservex[foundation]'",
        }
        _emit(payload, json_out=True)
        raise typer.Exit(3)

    try:
        import numpy as np
        from PIL import Image as _PIL
        from transformers import AutoImageProcessor, AutoModel  # type: ignore
        import torch  # type: ignore

        hf_repo = meta["hf_repo"]
        processor = AutoImageProcessor.from_pretrained(hf_repo)
        model = AutoModel.from_pretrained(hf_repo)
        model.eval()

        if device == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev = device
        model = model.to(dev)

        img = _PIL.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # DINOv2: CLS token is at index 0 of last_hidden_state
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, embedding)

        payload = {
            "status": "ok",
            "model_id": model_id,
            "hf_repo": hf_repo,
            "image": image,
            "embedding_shape": list(embedding.shape),
            "embedding_dim": int(embedding.shape[-1]),
            "saved_to": str(out),
            "device": dev,
        }
        _emit(payload, json_out=json_, explain=explain)

    except Exception as exc:
        payload = {
            "status": "error",
            "code": "EMBED_FAILED",
            "model_id": model_id,
            "image": image,
            "error": str(exc)[:500],
            "message": f"DINOv2 embedding failed: {exc!s:.300}",
        }
        _emit(payload, json_out=True)
        raise typer.Exit(1) from None


@app.command("knn")
def knn_cmd(
    model_id: str = typer.Argument(..., help="DINOv2 model ID."),
    query_image: str = typer.Argument(..., help="Query image path."),
    gallery_dir: Path = typer.Option(..., "--gallery-dir", help="Directory of gallery images."),
    top_k: int = typer.Option(5, "--top-k", help="Number of nearest neighbours to return."),
    device: str = typer.Option("auto", "--device"),
    json_: bool = typer.Option(False, "--json"),
    explain: bool = typer.Option(False, "--explain"),
) -> None:
    """kNN similarity search: find the top-k gallery images closest to a query image."""
    meta = _require_model(model_id)

    if explain:
        card = _status_card(model_id, meta)
        _emit(card, json_out=json_, explain=True)
        return

    if not meta.get("runnable", False) or meta["family"] != "dinov2":
        payload = {
            "status": "blocked",
            "code": meta.get("code") or "WRONG_TASK",
            "model_id": model_id,
            "message": "knn requires a runnable DINOv2 model (dinov2-small/base/large/giant).",
            "blocker": _blocker_for(meta),
            "fix": _fix_for({**meta, "model_id": model_id}),
        }
        _emit(payload, json_out=True)
        raise typer.Exit(3)

    query_path = Path(query_image)
    if not query_path.exists():
        _emit({"status": "error", "code": "INPUT_NOT_FOUND", "image": query_image,
               "message": f"Query image not found: {query_image}"}, json_out=True)
        raise typer.Exit(2)

    if not gallery_dir.exists() or not gallery_dir.is_dir():
        _emit({"status": "error", "code": "GALLERY_NOT_FOUND", "gallery_dir": str(gallery_dir),
               "message": f"Gallery directory not found: {gallery_dir}"}, json_out=True)
        raise typer.Exit(2)

    if not _probe("torch") or not _probe("transformers"):
        _emit({"status": "blocked", "code": "DEPS_MISSING",
               "message": "torch + transformers required. pip install 'visionservex[foundation]'"},
              json_out=True)
        raise typer.Exit(3)

    _img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    gallery_paths = [
        p for p in gallery_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _img_exts
    ]
    if not gallery_paths:
        _emit({"status": "error", "code": "GALLERY_EMPTY",
               "gallery_dir": str(gallery_dir),
               "message": "No images found in gallery directory."}, json_out=True)
        raise typer.Exit(2)

    try:
        import numpy as np
        from PIL import Image as _PIL
        from transformers import AutoImageProcessor, AutoModel  # type: ignore
        import torch  # type: ignore

        hf_repo = meta["hf_repo"]
        processor = AutoImageProcessor.from_pretrained(hf_repo)
        model_obj = AutoModel.from_pretrained(hf_repo)
        model_obj.eval()

        if device == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev = device
        model_obj = model_obj.to(dev)

        def _embed(img_path: Path) -> "np.ndarray":
            img = _PIL.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            with torch.no_grad():
                out = model_obj(**inputs)
            return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        query_emb = _embed(query_path)

        gallery_embs = []
        valid_paths = []
        for gp in gallery_paths:
            try:
                gallery_embs.append(_embed(gp))
                valid_paths.append(gp)
            except Exception:
                pass  # skip unreadable images silently

        if not gallery_embs:
            _emit({"status": "error", "code": "GALLERY_EMBED_FAILED",
                   "message": "Could not embed any gallery image."}, json_out=True)
            raise typer.Exit(1)

        gallery_matrix = np.stack(gallery_embs, axis=0)  # (N, D)
        # Cosine similarity
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        g_norms = gallery_matrix / (np.linalg.norm(gallery_matrix, axis=1, keepdims=True) + 1e-8)
        sims = (g_norms @ q_norm).tolist()

        ranked = sorted(
            enumerate(sims), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = [
            {
                "rank": i + 1,
                "image": str(valid_paths[idx]),
                "cosine_similarity": round(float(sim), 6),
            }
            for i, (idx, sim) in enumerate(ranked)
        ]

        payload = {
            "status": "ok",
            "model_id": model_id,
            "query_image": query_image,
            "gallery_dir": str(gallery_dir),
            "gallery_size": len(valid_paths),
            "top_k": top_k,
            "device": dev,
            "results": results,
        }
        _emit(payload, json_out=json_, explain=explain)

    except SystemExit:
        raise
    except Exception as exc:
        payload = {
            "status": "error",
            "code": "KNN_FAILED",
            "model_id": model_id,
            "query_image": query_image,
            "error": str(exc)[:500],
        }
        _emit(payload, json_out=True)
        raise typer.Exit(1) from None


@app.command("detect")
def detect_cmd(
    model_id: str = typer.Argument(..., help="GroundingDINO model ID."),
    image: str = typer.Argument(..., help="Path to input image."),
    text: str = typer.Option(..., "--text", help="Text prompt, e.g. 'cat . dog'."),
    out: Optional[Path] = typer.Option(None, "--out", help="Save detection JSON to this path."),
    threshold: float = typer.Option(0.3, "--threshold", help="Detection score threshold."),
    device: str = typer.Option("auto", "--device"),
    json_: bool = typer.Option(False, "--json"),
    explain: bool = typer.Option(False, "--explain"),
) -> None:
    """GroundingDINO text-prompted object detection. Saves boxes to --out (JSON)."""
    meta = _require_model(model_id)

    if explain:
        card = _status_card(model_id, meta)
        _emit(card, json_out=json_, explain=True)
        return

    if meta.get("api_required") or meta.get("auth_required"):
        payload = {
            "status": "blocked",
            "code": meta.get("code") or "API_REQUIRED",
            "model_id": model_id,
            "blocker": _blocker_for(meta),
            "fix": _fix_for({**meta, "model_id": model_id}),
            "message": (
                f"{model_id!r} requires API access. "
                f"Use `visionservex dino api {model_id} ...` for API-gated models."
            ),
        }
        _emit(payload, json_out=True)
        raise typer.Exit(3)

    if meta["family"] != "grounding-dino" or not meta.get("runnable", False):
        payload = {
            "status": "error",
            "code": "WRONG_TASK",
            "model_id": model_id,
            "message": (
                f"{model_id!r} is not a runnable GroundingDINO detection model. "
                "Use grounding-dino-swin-t, grounding-dino-swin-b, or grounding-dino-original-*."
            ),
        }
        _emit(payload, json_out=True)
        raise typer.Exit(2)

    image_path = Path(image)
    if not image_path.exists():
        _emit({"status": "error", "code": "INPUT_NOT_FOUND", "image": image,
               "message": f"Image not found: {image}"}, json_out=True)
        raise typer.Exit(2)

    if not _probe("torch"):
        _emit({"status": "blocked", "code": "TORCH_REQUIRED",
               "message": "PyTorch required. pip install torch",
               "fix": "pip install torch torchvision"}, json_out=True)
        raise typer.Exit(3)

    if not _probe("transformers"):
        _emit({"status": "blocked", "code": "TRANSFORMERS_REQUIRED",
               "message": "HuggingFace transformers required.",
               "fix": "pip install 'visionservex[open-vocab]'"}, json_out=True)
        raise typer.Exit(3)

    try:
        from PIL import Image as _PIL
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection  # type: ignore
        import torch  # type: ignore

        hf_repo = meta["hf_repo"]

        if device == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev = device

        processor = AutoProcessor.from_pretrained(hf_repo)
        model_obj = AutoModelForZeroShotObjectDetection.from_pretrained(hf_repo)
        model_obj.eval().to(dev)

        img = _PIL.open(image_path).convert("RGB")
        inputs = processor(images=img, text=text, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model_obj(**inputs)

        results_raw = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=threshold,
            text_threshold=threshold,
            target_sizes=[img.size[::-1]],
        )

        detections = []
        for result in results_raw:
            for box, score, label in zip(
                result["boxes"].tolist(),
                result["scores"].tolist(),
                result["labels"],
            ):
                detections.append({
                    "box_xyxy": [round(c, 2) for c in box],
                    "score": round(float(score), 4),
                    "label": label,
                })

        payload = {
            "status": "ok",
            "model_id": model_id,
            "hf_repo": hf_repo,
            "image": image,
            "text_prompt": text,
            "threshold": threshold,
            "device": dev,
            "n_detections": len(detections),
            "detections": detections,
        }

        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2))
            payload["saved_to"] = str(out)

        _emit(payload, json_out=json_, explain=explain)

    except SystemExit:
        raise
    except Exception as exc:
        payload = {
            "status": "error",
            "code": "DETECT_FAILED",
            "model_id": model_id,
            "image": image,
            "error": str(exc)[:500],
            "message": f"GroundingDINO detection failed: {exc!s:.300}",
        }
        _emit(payload, json_out=True)
        raise typer.Exit(1) from None


@app.command("api")
def api_cmd(
    model_id: str = typer.Argument(..., help="API-only model ID (e.g. dino-x-api, grounding-dino-1.5)."),
    image: str = typer.Argument(..., help="Path to input image."),
    text: str = typer.Option(..., "--text", help="Text prompt."),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="DEEPDATASPACE_API_KEY",
        help="API key (or set $DEEPDATASPACE_API_KEY). Never logged in full.",
    ),
    json_: bool = typer.Option(False, "--json"),
    explain: bool = typer.Option(False, "--explain"),
) -> None:
    """For API-only models: if key absent, return auth_required JSON; never log key in full."""
    meta = _require_model(model_id)

    if explain:
        card = _status_card(model_id, meta)
        _emit(card, json_out=json_, explain=True)
        return

    if not meta.get("api_required", False):
        payload = {
            "status": "error",
            "code": "NOT_AN_API_MODEL",
            "model_id": model_id,
            "message": (
                f"{model_id!r} does not require an external API key. "
                "Use `visionservex dino embed` or `visionservex dino detect` instead."
            ),
        }
        _emit(payload, json_out=True)
        raise typer.Exit(2)

    # Also check env var directly as a fallback
    env_var = meta.get("env_var", "DEEPDATASPACE_API_KEY")
    resolved_key = api_key or os.environ.get(env_var, "")

    if not resolved_key:
        payload = {
            "status": "auth_required",
            "code": "API_KEY_MISSING",
            "model_id": model_id,
            "family": meta["family"],
            "task": meta["task"],
            "license": meta["license"],
            "auth_required": True,
            "api_required": True,
            "env_var": env_var,
            "blocker": f"API key required — set ${env_var}",
            "fix": (
                f"export {env_var}=<your-key> && "
                f"visionservex dino api {model_id} {image} --text '{text}'"
            ),
            "message": (
                f"{model_id!r} is an API-only model. "
                f"Set ${env_var} to call the external endpoint."
            ),
        }
        _emit(payload, json_out=json_, explain=explain)
        raise typer.Exit(3)

    # Key is present — show redacted version, never full key
    redacted = _redact_key(resolved_key)

    # At this point we would call the external API, but VisionServeX does not
    # mirror proprietary weights or proxy closed API endpoints. Return a
    # structured next-step instead of silently proxying.
    payload = {
        "status": "api_key_present",
        "code": "EXTERNAL_API_CALL_REQUIRED",
        "model_id": model_id,
        "family": meta["family"],
        "task": meta["task"],
        "license": meta["license"],
        "auth_required": True,
        "api_required": True,
        "api_key_provided": True,
        "api_key_redacted": redacted,
        "env_var": env_var,
        "image": image,
        "text_prompt": text,
        "message": (
            f"API key accepted (shown redacted: {redacted}). "
            f"{model_id!r} calls an external closed endpoint. "
            "VisionServeX surfaces the status and key-check only; "
            "call the DeepDataSpace API directly with your key."
        ),
        "next_step": (
            f"curl -X POST https://api.deepdataspace.com/v1/detect "
            f"-H 'Authorization: Bearer <{env_var}>' "
            f"-F 'image=@{image}' -F 'text={text}'"
        ),
        "docs": "https://deepdataspace.com/docs",
    }
    _emit(payload, json_out=json_, explain=explain)


__all__ = ["app"]
