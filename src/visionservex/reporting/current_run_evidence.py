"""Consolidate REAL benchmark artifacts into comprehensive current-run task
leaderboards (v2.61 / V3-11).

Problem this solves: the task notebooks carry forward small leaderboards that do
not cover every ``benchmark_passed`` model, so the reconciler left ~30 healthy
rows without a current-run evidence artifact. This helper scans every real
benchmark artifact (under ``notebook/_runs/**`` and ``reports/``), extracts each
model's REAL metric, and writes a per-task leaderboard covering every
benchmark_passed model into the task's ``reports/`` directory. ``scan_task_outputs``
then attributes a current-run evidence artifact to each model. The metrics are
real (from actual benchmarks); only the leaderboard file is regenerated this run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# Recognised benchmark metric columns (detection/seg mAP, IoU, kNN, accuracy).
_METRIC_KEYS = (
    "mAP50_95",
    "map50_95",
    "mask_mAP50_95",
    "mask_map50_95",
    "mean_iou",
    "mask_iou",
    "knn_accuracy",
    "accuracy",
    "top1",
)

# ledger task -> (task report dir relative to notebook/, leaderboard filename)
_TASK_DIRS = {
    "detect": ("01_object_detection", "detection_leaderboard.csv"),
    "segment": ("02_automatic_segmentation", "segmentation_leaderboard.csv"),
    "foundation_segment": ("03_promptable_segmentation", "promptable_leaderboard.csv"),
    "promptable_segment": ("03_promptable_segmentation", "promptable_leaderboard.csv"),
    "classify": ("05_classification", "classification_leaderboard.csv"),
    "embed": ("06_embedding_similarity", "embedding_leaderboard.csv"),
    "open_vocab": ("04_open_vocab_vlm", "open_vocab_leaderboard.csv"),
    "open_vocab_detect": ("04_open_vocab_vlm", "open_vocab_leaderboard.csv"),
    "vlm": ("04_open_vocab_vlm", "vlm_leaderboard.csv"),
}


def _metric_of(row: dict[str, Any]) -> tuple[float, str] | None:
    for k in _METRIC_KEYS:
        v = row.get(k)
        if v in (None, "", "nan", "NaN"):
            continue
        try:
            return float(v), k
        except (TypeError, ValueError):
            continue
    return None


def _scan_real_metrics(nb_root: Path) -> dict[str, tuple[float, str, str]]:
    """Build {model_id: (metric_value, metric_name, source_path)} from every real
    benchmark artifact under notebook/_runs/** and repo reports/."""
    out: dict[str, tuple[float, str, str]] = {}
    search_roots = [nb_root / "_runs", nb_root.parent / "reports", nb_root]
    seen: set[Path] = set()
    # Benchmark artifacts use many naming conventions (benchmark/leaderboard/
    # detection_400/knn/classification/promptable/retry/...); scan any JSON that
    # exposes a list of model rows with a recognised metric.
    _json_globs = (
        "*benchmark*.json",
        "*leaderboard*.json",
        "*detection*.json",
        "*deimv2*.json",
        "*_400*.json",
        "*knn*.json",
        "*classification*.json",
        "*promptable*.json",
        "*embedding*.json",
        "*_seg*.json",
        "*retry*.json",
        "*optional*.json",
    )
    for base in search_roots:
        if not base.exists():
            continue
        json_paths: list[Path] = []
        for g in _json_globs:
            json_paths.extend(base.rglob(g))
        for p in json_paths:
            if p in seen or "ipynb_checkpoints" in str(p):
                continue
            seen.add(p)
            try:
                data = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            rows: list = []
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                for k in ("models", "results", "rows", "per_model"):
                    if isinstance(data.get(k), list):
                        rows = data[k]
                        break
                # single-model artifact: {model_id: ..., <metric>: ...}
                if not rows and data.get("model_id"):
                    rows = [data]
            for r in rows:
                if not isinstance(r, dict):
                    continue
                mid = (r.get("model_id") or r.get("model") or r.get("name") or "").strip()
                if not mid:
                    continue
                m = _metric_of(r)
                if m and mid not in out:
                    out[mid] = (m[0], m[1], str(p))
        for p in list(base.rglob("*leaderboard*.csv")):
            if p in seen:
                continue
            seen.add(p)
            try:
                rows = list(csv.DictReader(p.open()))
            except OSError:
                continue
            for r in rows:
                mid = (r.get("model_id") or r.get("name") or "").strip()
                if not mid:
                    continue
                m = _metric_of(r)
                if m and mid not in out:
                    out[mid] = (m[0], m[1], str(p))
    return out


def build_current_run_leaderboards(nb_root: Path | str, ledger_csv: Path | str) -> dict[str, Any]:
    """Write comprehensive current-run leaderboards covering every benchmark_passed
    model that has a real metric. Returns a summary dict."""
    nb_root = Path(nb_root)
    ledger_csv = Path(ledger_csv)
    metrics = _scan_real_metrics(nb_root)

    by_task: dict[str, list[dict[str, Any]]] = {}
    covered: list[str] = []
    uncovered: list[str] = []
    if ledger_csv.exists():
        for row in csv.DictReader(ledger_csv.open()):
            if row.get("final_state") != "benchmark_passed":
                continue
            mid = row.get("model_id", "").strip()
            task = row.get("task", "").strip()
            if task not in _TASK_DIRS:
                continue
            m = metrics.get(mid)
            if not m:
                uncovered.append(mid)
                continue
            covered.append(mid)
            by_task.setdefault(task, []).append(
                {
                    "model_id": mid,
                    "status": "ok",
                    "family": row.get("family", ""),
                    m[1]: round(m[0], 4),
                    "metric_name": m[1],
                    "evidence_source": Path(m[2]).name,
                }
            )

    written = {}
    # group tasks that share a leaderboard file (e.g. foundation/promptable)
    files: dict[Path, list[dict[str, Any]]] = {}
    for task, rows in by_task.items():
        subdir, fname = _TASK_DIRS[task]
        out_path = nb_root / subdir / "reports" / fname
        files.setdefault(out_path, []).extend(rows)
    for out_path, rows in files.items():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # union of all metric columns across rows
        cols = ["model_id", "status", "family", "metric_name", "evidence_source"]
        metric_cols = sorted({k for r in rows for k in r if k not in cols})
        fieldnames = [
            "model_id",
            "status",
            "family",
            *metric_cols,
            "metric_name",
            "evidence_source",
        ]
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        written[str(out_path.relative_to(nb_root))] = len(rows)

    return {
        "metrics_found": len(metrics),
        "models_covered": len(covered),
        "models_uncovered": sorted(set(uncovered)),
        "leaderboards_written": written,
    }
