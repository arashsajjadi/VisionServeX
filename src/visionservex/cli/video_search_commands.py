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
    tracker: str = typer.Option(
        "simple-iou",
        "--tracker",
        help="Tracker backend: simple-iou (default), bytetrack, bot-sort, ocsort.",
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Index a video or frame folder for later text retrieval."""
    _print_privacy()

    from visionservex import VisionModel
    from visionservex.runtime.trackers import TrackerUnavailableError, build_tracker
    from visionservex.runtime.video_search import (
        build_index,
        detections_from_result,
        embedding_from_result,
    )

    if not source.exists():
        console.print(f"[red]Source not found:[/red] {source}")
        raise typer.Exit(2)

    # Build tracker adapter (raises TrackerUnavailableError if package missing)
    try:
        tracker_adapter = build_tracker(tracker)
    except TrackerUnavailableError as exc:
        payload = exc.to_dict()
        if json_:
            print(json.dumps(payload, indent=2))
        else:
            console.print(f"[red]{exc.code}[/red]: {exc.name} not installed")
            console.print(f"  install: {exc.install}")
        raise typer.Exit(3)

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
        tracker_instance=tracker_adapter,
        tracker_name=tracker,
    )

    summary = {
        "index_dir": str(out_path),
        "detector": detector,
        "embedder": embedder,
        "tracker": tracker,
    }
    if json_:
        print(json.dumps(summary, indent=2))
        return
    console.print(f"[green]Index written to[/green] {out_path}")
    console.print(f"  tracker: {tracker}")


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


# ---------------------------------------------------------------------------
# trackers — list available tracker backends
# ---------------------------------------------------------------------------

_TRACKER_REGISTRY = {
    "simple-iou": {
        "installed": True,
        "description": "SimpleIoU: built-in tracker (IoU-based, no extra install).",
        "install": None,
    },
    "bytetrack": {
        "installed": False,
        "description": "ByteTrack: multi-object tracker (BYTE algorithm). Apache-2.0.",
        "install": "pip install bytetracker  # or: git clone https://github.com/ifzhang/ByteTrack",
        "blocker": "BYTETRACK_REQUIRED",
    },
    "bot-sort": {
        "installed": False,
        "description": "BoT-SORT: robust multi-object tracker. Apache-2.0.",
        "install": "git clone https://github.com/NirAharon/BoT-SORT && pip install -e .",
        "blocker": "BOTSORT_REQUIRED",
    },
    "ocsort": {
        "installed": False,
        "description": "OC-SORT: observation-centric SORT. MIT.",
        "install": "pip install ocsort",
        "blocker": "OCSORT_REQUIRED",
    },
}

_REID_REGISTRY = {
    "cosine-siglip2": {
        "installed": True,
        "description": "Built-in: cosine similarity on SigLIP2 embeddings.",
        "install": None,
    },
    "osnet": {
        "installed": False,
        "description": "OSNet (Torchreid): lightweight person ReID backbone. MIT.",
        "install": "pip install torchreid  # or: pip install git+https://github.com/KaiyangZhou/deep-person-reid",
        "blocker": "TORCHREID_REQUIRED",
    },
    "fastreid": {
        "installed": False,
        "description": "FastReID: strong baseline for person ReID. Apache-2.0.",
        "install": "git clone https://github.com/JDAI-CV/fast-reid && pip install -e .",
        "blocker": "FASTREID_REQUIRED",
    },
}


def _probe_tracker(name: str) -> bool:
    """Return True if the tracker package can be imported."""
    pkg_map = {"bytetrack": "bytetracker", "bot-sort": "botsort", "ocsort": "ocsort"}
    pkg = pkg_map.get(name)
    if not pkg:
        return False
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def _probe_reid(name: str) -> bool:
    pkg_map = {"osnet": "torchreid", "fastreid": "fastreid"}
    pkg = pkg_map.get(name)
    if not pkg:
        return False
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


@app.command("trackers")
def list_trackers(json_: bool = typer.Option(False, "--json")) -> None:
    """List available tracker backends and their install status."""
    results = {}
    for name, info in _TRACKER_REGISTRY.items():
        installed = info["installed"] or _probe_tracker(name)
        results[name] = {**info, "installed": installed}

    if json_:
        print(json.dumps(results, indent=2))
        return

    table = Table(show_header=True)
    table.add_column("Tracker")
    table.add_column("Installed")
    table.add_column("Description")
    for name, info in results.items():
        status = "[green]yes[/green]" if info["installed"] else "[red]no[/red]"
        table.add_row(name, status, info["description"])
    console.print(table)


@app.command("reid-models")
def list_reid(json_: bool = typer.Option(False, "--json")) -> None:
    """List available ReID model backends and their install status."""
    results = {}
    for name, info in _REID_REGISTRY.items():
        installed = info["installed"] or _probe_reid(name)
        results[name] = {**info, "installed": installed}

    if json_:
        print(json.dumps(results, indent=2))
        return

    table = Table(show_header=True)
    table.add_column("ReID Backend")
    table.add_column("Installed")
    table.add_column("Description")
    for name, info in results.items():
        status = "[green]yes[/green]" if info["installed"] else "[red]no[/red]"
        table.add_row(name, status, info["description"])
    console.print(table)


@app.command("doctor")
def doctor_cmd(
    tracker: str = typer.Option(
        "", "--tracker", help="Check tracker: bytetrack, bot-sort, ocsort."
    ),
    reid: str = typer.Option("", "--reid", help="Check ReID backend: osnet, fastreid."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Check tracker/ReID backend availability and print install command if missing."""
    results: dict = {}

    if tracker:
        info = _TRACKER_REGISTRY.get(tracker)
        if not info:
            available = list(_TRACKER_REGISTRY)
            payload = {"code": "TRACKER_UNKNOWN", "tracker": tracker, "available": available}
            print(json.dumps(payload, indent=2)) if json_ else console.print(
                f"[red]Unknown tracker:[/red] {tracker}"
            )
            raise typer.Exit(2)
        installed = info["installed"] or _probe_tracker(tracker)
        entry = {
            "tracker": tracker,
            "installed": installed,
            "description": info["description"],
        }
        if not installed:
            entry["code"] = info.get("blocker", "TRACKER_REQUIRED")
            entry["install"] = info.get("install", "")
        results["tracker"] = entry

    if reid:
        info = _REID_REGISTRY.get(reid)
        if not info:
            available = list(_REID_REGISTRY)
            payload = {"code": "REID_UNKNOWN", "reid": reid, "available": available}
            print(json.dumps(payload, indent=2)) if json_ else console.print(
                f"[red]Unknown ReID:[/red] {reid}"
            )
            raise typer.Exit(2)
        installed = info["installed"] or _probe_reid(reid)
        entry = {
            "reid": reid,
            "installed": installed,
            "description": info["description"],
        }
        if not installed:
            entry["code"] = info.get("blocker", "REID_REQUIRED")
            entry["install"] = info.get("install", "")
        results["reid"] = entry

    if not tracker and not reid:
        results["message"] = (
            "Specify --tracker or --reid. Use 'visionservex video-search trackers' to list options."
        )

    if json_:
        print(json.dumps(results, indent=2))
        return

    for key, val in results.items():
        if isinstance(val, dict):
            label = val.get("tracker") or val.get("reid") or key
            status = (
                "[green]installed[/green]" if val.get("installed") else "[red]not installed[/red]"
            )
            console.print(f"  {label}: {status}")
            if not val.get("installed") and "install" in val:
                console.print(f"  install: [cyan]{val['install']}[/cyan]")
        else:
            console.print(val)
