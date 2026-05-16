# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Surveillance / appearance-based video search CLI.

Local-only. No face recognition. No biometric identity.
See PRIVACY_NOTICE in runtime/video_search.py.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="Surveillance / appearance-based video search (local-only).",
    no_args_is_help=True,
)
console = Console()


def _print_privacy() -> None:
    from visionservex.runtime.video_search import PRIVACY_NOTICE

    console.print(f"[yellow]Privacy:[/yellow] {PRIVACY_NOTICE}\n")


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


@app.command("index")
def index_cmd(
    source: Path = typer.Argument(..., help="Video file OR folder of frames."),
    out: Path = typer.Option(..., "--out", help="Output index directory."),
    detector: str = typer.Option(
        "owlv2-base-patch16",
        "--detector",
        help="VisionModel detector with prompt support (owlv2-*, grounding-dino-*).",
    ),
    embedder: str = typer.Option(
        "siglip2-base-patch16-224",
        "--embedder",
        help="VisionModel embedder (siglip2-*, dinov2-*).",
    ),
    prompt: str = typer.Option("person", "--prompt", help="Detection prompt (free-form text)."),
    sample_fps: float = typer.Option(1.0, "--sample-fps", help="Frames per second to sample."),
    stride: int = typer.Option(0, "--stride", help="Frame stride; overrides --sample-fps."),
    max_frames: int = typer.Option(
        0, "--max-frames", help="Hard cap on number of frames to ingest (0 = no cap)."
    ),
    threshold: float = typer.Option(0.1, "--threshold", help="Detector confidence threshold."),
    auto_pull: bool = typer.Option(False, "--auto-pull", help="Allow checkpoint auto-download."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Index a video or frame folder for later text retrieval."""
    _print_privacy()

    from visionservex import VisionModel
    from visionservex.runtime.video_search import (
        build_index,
        detections_from_result,
        embedding_from_result,
    )

    if not source.exists():
        console.print(f"[red]Source not found:[/red] {source}")
        raise typer.Exit(2)

    # Build detector + embedder via VisionModel.
    det_model = VisionModel(detector, auto_pull=auto_pull)
    emb_model = VisionModel(embedder, auto_pull=auto_pull)

    def detect_fn(image, p):
        result = det_model.predict(image, prompt=p, threshold=threshold)
        return detections_from_result(result)

    def embed_fn(crop):
        result = emb_model.predict(crop)
        return embedding_from_result(result)

    out_path = build_index(
        source=str(source),
        out_dir=str(out),
        detect_fn=detect_fn,
        embed_fn=embed_fn,
        detector_model_id=detector,
        embedder_model_id=embedder,
        prompt=prompt,
        sample_fps=sample_fps if sample_fps > 0 else None,
        stride=stride if stride > 0 else None,
        max_frames=max_frames if max_frames > 0 else None,
    )

    summary = {"index_dir": str(out_path), "detector": detector, "embedder": embedder}
    if json_:
        print(json.dumps(summary, indent=2))
        return
    console.print(f"[green]Index written to[/green] {out_path}")


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


@app.command("query")
def query_cmd(
    index_dir: Path = typer.Argument(..., help="Index directory created by `index`."),
    text: str = typer.Option(..., "--text", help="Free-form query text."),
    top_k: int = typer.Option(20, "--top-k"),
    embedder: str = typer.Option(
        "",
        "--embedder",
        help="Text-capable embedder. Defaults to the one used at index time.",
    ),
    out: Path = typer.Option(None, "--out", help="Write HTML timeline to this path."),
    show_timeline: bool = typer.Option(False, "--show-timeline"),
    auto_pull: bool = typer.Option(False, "--auto-pull"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Query an index with a text prompt and return ranked crops."""
    _print_privacy()

    from visionservex import VisionModel
    from visionservex.runtime.video_search import (
        embedding_from_result,
        load_index,
        query_index,
        render_timeline_html,
    )

    manifest, _ = load_index(index_dir)
    text_embedder = embedder or manifest.embedder_model_id
    if not text_embedder:
        console.print("[red]No embedder specified and none recorded in index.[/red]")
        raise typer.Exit(2)

    # Build text embedding via the embedder's predict() — SigLIP2/CLIP-class
    # models support text via their processor. If a model doesn't, the caller
    # will see a clear engine error.
    emb_model = VisionModel(text_embedder, auto_pull=auto_pull)
    # Most embed engines accept a text= kwarg through metadata. We pass the
    # raw text image as a placeholder when no text-only path is available;
    # production code should provide a model that supports text embeddings.
    text_result = emb_model.predict(text, text=text)  # type: ignore[arg-type]
    query_vec = embedding_from_result(text_result)

    hits = query_index(index_dir, query_vec, top_k=top_k)
    payload = {
        "query": text,
        "index_dir": str(index_dir),
        "embedder": text_embedder,
        "hits": [h.to_dict() for h in hits],
    }

    if out:
        html = render_timeline_html(hits, query_text=text, source=manifest.source)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)

    if json_:
        print(json.dumps(payload, indent=2))
        return

    table = Table(title=f"video-search: {text!r}", show_header=True)
    table.add_column("Rank", no_wrap=True)
    table.add_column("Time", no_wrap=True)
    table.add_column("Track", no_wrap=True)
    table.add_column("Label")
    table.add_column("Sim", no_wrap=True)
    for i, h in enumerate(hits, 1):
        table.add_row(
            str(i), f"{h.timestamp_s:.2f}s", str(h.track_id), h.label, f"{h.similarity:.3f}"
        )
    console.print(table)
    if show_timeline and out is None:
        console.print("[dim]Pass --out path.html to save a timeline.[/dim]")


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@app.command("inspect")
def inspect_cmd(
    index_dir: Path = typer.Argument(..., help="Index directory."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Print a summary of an existing index."""
    from visionservex.runtime.video_search import load_index

    manifest, emb = load_index(index_dir)
    summary = {
        "index_dir": str(index_dir),
        "detector": manifest.detector_model_id,
        "embedder": manifest.embedder_model_id,
        "tracker": manifest.tracker,
        "source": manifest.source,
        "n_frames_seen": manifest.n_frames_seen,
        "n_detections": manifest.n_detections,
        "n_tracks": manifest.n_tracks,
        "embedding_dim": int(manifest.embedding_dim),
        "embedding_rows": int(emb.shape[0]) if hasattr(emb, "shape") else len(emb),
    }
    if json_:
        print(json.dumps(summary, indent=2))
        return
    table = Table(title=f"index summary — {index_dir}", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for k, v in summary.items():
        table.add_row(k, str(v))
    console.print(table)


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------


@app.command("cleanup")
def cleanup_cmd(
    index_dir: Path = typer.Argument(..., help="Index directory to delete."),
    confirm: bool = typer.Option(False, "--yes", help="Skip confirmation."),
) -> None:
    """Delete an index directory (local cleanup)."""
    if not index_dir.exists():
        console.print(f"[dim]{index_dir} does not exist.[/dim]")
        return
    if not confirm:
        console.print(f"[red]This will delete {index_dir}. Re-run with --yes to confirm.[/red]")
        raise typer.Exit(1)
    shutil.rmtree(index_dir, ignore_errors=True)
    console.print(f"[green]Removed[/green] {index_dir}")
