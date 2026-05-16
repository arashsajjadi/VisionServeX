# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Embedding / feature-intelligence CLI commands.

Powered by DINOv2 / SigLIP2 / other feature backbones.

Commands:
- embed MODEL image_or_folder
- similarity MODEL image1 image2
- index MODEL folder/
- search MODEL query_image
- deduplicate MODEL folder/
- active-select MODEL folder/
- domain-shift MODEL train/ test/
- dataset-report MODEL folder/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="Feature-intelligence: embed, similarity, index, search, dedup.")


@app.command("embed", help="Compute embeddings for an image or folder.")
def embed_cmd(
    model_id: str,
    target: Path = typer.Argument(..., help="Image file or folder."),
    out: Path | None = typer.Option(None, "--out", help="Output .npy (single) or folder (batch)."),
    device: str = typer.Option("auto", "--device"),
    max_images: int | None = typer.Option(None, "--max-images"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Compute embeddings. Output: .npy file (single image) or batch dir (folder)."""
    from visionservex.core.embedding_results import EmbeddingResult
    from visionservex.core.model import VisionModel

    if target.is_file():
        from PIL import Image as _PIL

        with VisionModel(model_id, device=device) as model:
            result = model.predict(_PIL.open(target).convert("RGB"))
        if not isinstance(result, EmbeddingResult):
            console.print(f"[red]error:[/red] model {model_id} does not return embeddings")
            raise typer.Exit(1)
        payload = result.to_dict()
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, result.embedding)
            payload["saved_to"] = str(out)
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(
                f"[bold]Embedding:[/bold] {model_id}  "
                f"dim={result.embedding_dim}  "
                f"norm={np.linalg.norm(result.embedding):.3f}  "
                f"latency={result.latency_ms:.1f}ms"
            )
            if out:
                console.print(f"  Saved to: {out}")
        return

    if target.is_dir():
        from visionservex.runtime.embeddings import embed_folder

        if not json_:
            console.print(f"[bold]Embedding folder:[/bold] {target} ({model_id})")
        embeddings, paths = embed_folder(model_id, target, device=device, max_images=max_images)
        payload = {
            "model_id": model_id,
            "folder": str(target),
            "n_images": embeddings.shape[0],
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        }
        if out:
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "embeddings.npy", embeddings)
            (out / "image_paths.json").write_text(json.dumps(paths, indent=2), encoding="utf-8")
            payload["saved_to"] = str(out)
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(
                f"  Embedded {payload['n_images']} images  dim={payload['embedding_dim']}"
            )
            if out:
                console.print(f"  Saved to: {out}/embeddings.npy + image_paths.json")
        return

    console.print(f"[red]error:[/red] target {target} is neither a file nor a folder")
    raise typer.Exit(1)


@app.command("similarity", help="Cosine similarity between two images.")
def similarity_cmd(
    model_id: str,
    image_a: Path,
    image_b: Path,
    device: str = typer.Option("auto", "--device"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from PIL import Image as _PIL

    from visionservex.core.embedding_results import EmbeddingResult, SimilarityResult
    from visionservex.core.model import VisionModel
    from visionservex.runtime.embeddings import cosine_similarity

    with VisionModel(model_id, device=device) as model:
        r_a = model.predict(_PIL.open(image_a).convert("RGB"))
        r_b = model.predict(_PIL.open(image_b).convert("RGB"))

    if not (isinstance(r_a, EmbeddingResult) and isinstance(r_b, EmbeddingResult)):
        console.print(f"[red]error:[/red] {model_id} does not return embeddings")
        raise typer.Exit(1)

    sim = cosine_similarity(r_a.embedding, r_b.embedding)
    result = SimilarityResult(
        model_id=model_id,
        image_a=str(image_a),
        image_b=str(image_b),
        cosine_similarity=round(sim, 6),
        embedding_dim=r_a.embedding_dim,
    )
    if json_:
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        console.print(
            f"[bold]Cosine similarity:[/bold] {sim:.4f}  "
            f"(model={model_id}, dim={r_a.embedding_dim})"
        )


@app.command("index", help="Build a search index from a folder of images.")
def index_cmd(
    model_id: str,
    folder: Path,
    out: Path = typer.Option(..., "--out", help="Output index directory."),
    device: str = typer.Option("auto", "--device"),
    max_images: int | None = typer.Option(None, "--max-images"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.runtime.embeddings import EmbeddingIndex, embed_folder

    if not folder.exists():
        console.print(f"[red]error:[/red] folder not found: {folder}")
        raise typer.Exit(1)

    if not json_:
        console.print(f"[bold]Building index:[/bold] {folder} → {out}")
    embeddings, paths = embed_folder(model_id, folder, device=device, max_images=max_images)
    index = EmbeddingIndex(
        index_dir=out, embeddings=embeddings, image_paths=paths, model_id=model_id
    )
    index.save()
    payload = {
        "model_id": model_id,
        "index_dir": str(out),
        "n_images": len(paths),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
    }
    if json_:
        typer.echo(json.dumps(payload, indent=2))
    else:
        console.print(
            f"[green]Index built.[/green] {payload['n_images']} images, "
            f"dim={payload['embedding_dim']}  →  {out}/"
        )


@app.command("search", help="Search an index for nearest neighbors of a query image.")
def search_cmd(
    model_id: str,
    query: Path,
    index_dir: Path = typer.Option(
        ..., "--index", help="Index directory built by 'index' command."
    ),
    top_k: int = typer.Option(10, "--top-k"),
    device: str = typer.Option("auto", "--device"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from PIL import Image as _PIL

    from visionservex.core.embedding_results import EmbeddingResult, SearchHit, SearchResult
    from visionservex.core.model import VisionModel
    from visionservex.runtime.embeddings import EmbeddingIndex, search_index

    index = EmbeddingIndex.load(index_dir)

    with VisionModel(model_id, device=device) as model:
        r = model.predict(_PIL.open(query).convert("RGB"))
    if not isinstance(r, EmbeddingResult):
        console.print(f"[red]error:[/red] {model_id} does not return embeddings")
        raise typer.Exit(1)

    hits_raw = search_index(index, r.embedding, top_k=top_k)
    hits = [SearchHit(image_path=p, score=s, rank=i + 1) for i, (p, s) in enumerate(hits_raw)]
    result = SearchResult(
        model_id=model_id,
        query=str(query),
        top_k=top_k,
        hits=hits,
        index_path=str(index_dir),
    )
    if json_:
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        console.print(f"[bold]Top-{top_k} hits for[/bold] {query}")
        table = Table(show_lines=False)
        for col in ("Rank", "Score", "Image"):
            table.add_column(col)
        for h in hits:
            table.add_row(str(h.rank), f"{h.score:.4f}", h.image_path)
        console.print(table)


@app.command("deduplicate", help="Find likely-duplicate images in a folder.")
def deduplicate_cmd(
    model_id: str,
    folder: Path,
    threshold: float = typer.Option(0.98, "--threshold", min=0.5, max=1.0),
    out: Path | None = typer.Option(None, "--out"),
    device: str = typer.Option("auto", "--device"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.runtime.embeddings import EmbeddingIndex, deduplicate_index, embed_folder

    if not json_:
        console.print(f"[bold]Deduplicating:[/bold] {folder} threshold={threshold}")
    embeddings, paths = embed_folder(model_id, folder, device=device)
    index = EmbeddingIndex(
        index_dir=Path("/tmp/visionservex_dedup_index"),
        embeddings=embeddings,
        image_paths=paths,
        model_id=model_id,
    )
    pairs = deduplicate_index(index, threshold=threshold)

    payload = {
        "model_id": model_id,
        "folder": str(folder),
        "threshold": threshold,
        "n_images": len(paths),
        "n_likely_duplicates": len(pairs),
        "pairs": [{"a": a, "b": b, "score": s} for a, b, s in pairs[:50]],
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".csv":
            import csv

            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["image_a", "image_b", "similarity"])
                for a, b, s in pairs:
                    w.writerow([a, b, f"{s:.6f}"])
        else:
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["saved_to"] = str(out)

    if json_:
        typer.echo(json.dumps(payload, indent=2))
    else:
        console.print(f"  Found {len(pairs)} likely-duplicate pairs (sim ≥ {threshold})")
        for a, b, s in pairs[:10]:
            console.print(f"  [dim]{s:.4f}[/dim]  {a}  ↔  {b}")
        if out:
            console.print(f"\n  Saved to {out}")


@app.command("dataset-report", help="Generate a dataset report from a folder.")
def dataset_report_cmd(
    model_id: str,
    folder: Path,
    out: Path | None = typer.Option(None, "--out"),
    device: str = typer.Option("auto", "--device"),
    max_images: int | None = typer.Option(None, "--max-images"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.runtime.embeddings import build_dataset_report

    if not json_:
        console.print(f"[bold]Dataset report:[/bold] {folder} ({model_id})")
    report = build_dataset_report(model_id, folder, max_images=max_images, device=device)

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".md":
            out.write_text(report.to_markdown(), encoding="utf-8")
        else:
            out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if json_:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        console.print(report.to_markdown())
        if out:
            console.print(f"\n[green]Saved to {out}[/green]")


@app.command("active-select", help="Select diverse samples for active learning.")
def active_select_cmd(
    model_id: str,
    folder: Path,
    budget: int = typer.Option(100, "--budget"),
    out: Path | None = typer.Option(None, "--out"),
    device: str = typer.Option("auto", "--device"),
    max_images: int | None = typer.Option(None, "--max-images"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.runtime.embeddings import active_learning_select

    selected = active_learning_select(
        model_id, folder, budget=budget, device=device, max_images=max_images
    )
    payload = {
        "model_id": model_id,
        "folder": str(folder),
        "budget": budget,
        "n_selected": len(selected),
        "selected": selected,
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".csv":
            import csv

            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["image_path"])
                for p in selected:
                    w.writerow([p])
        else:
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["saved_to"] = str(out)
    if json_:
        typer.echo(json.dumps(payload, indent=2))
    else:
        console.print(f"[bold]Selected {len(selected)} samples (budget={budget})[/bold]")
        for p in selected[:20]:
            console.print(f"  {p}")
        if len(selected) > 20:
            console.print(f"  ... and {len(selected) - 20} more")
        if out:
            console.print(f"\n[green]Saved to {out}[/green]")


@app.command("domain-shift", help="Estimate domain shift between train and test folders.")
def domain_shift_cmd(
    model_id: str,
    train_folder: Path,
    test_folder: Path,
    out: Path | None = typer.Option(None, "--out"),
    device: str = typer.Option("auto", "--device"),
    max_images: int = typer.Option(200, "--max-images"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.runtime.embeddings import domain_shift_report

    if not json_:
        console.print(f"[bold]Domain shift:[/bold] {train_folder} → {test_folder}")
    report = domain_shift_report(
        model_id, train_folder, test_folder, device=device, max_images=max_images
    )
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if json_:
        typer.echo(json.dumps(report, indent=2))
    else:
        if report["status"] == "ok":
            console.print(f"  Centroid similarity:        {report['centroid_similarity']:.4f}")
            console.print(
                f"  Test→Train mean similarity: {report['test_to_train_mean_nearest_similarity']:.4f}"
            )
            console.print(f"  Domain shift score:         {report['domain_shift_score']:.4f}")
            console.print(f"  Verdict:                    [bold]{report['verdict']}[/bold]")
            console.print(f"\n  [dim]{report['note']}[/dim]")
        else:
            console.print(f"[yellow]{report['status']}:[/yellow] {report.get('reason', '')}")


__all__ = ["app"]
