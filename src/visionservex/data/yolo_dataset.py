# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""YOLO dataset (``data.yaml``) resolution and validation for training.

LibreYOLO training consumes a *YOLO-format* dataset described by a ``data.yaml``
that declares the dataset root, the ``train``/``val`` image splits, the class
count (``nc``) and the class ``names``. This module validates that contract and
resolves a user-supplied path (a YAML file *or* a directory containing one) to a
concrete ``data.yaml``.

Safety:
- Parsing uses ``yaml.safe_load`` only — it can never execute embedded Python.
- A ``download:`` block (Ultralytics-style auto-download script) is *reported*
  but **never executed** here. LibreYOLO training is invoked with
  ``allow_download_scripts=False``.
- This module imports neither ``ultralytics`` nor any AGPL/GPL runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_YAML_EXTS = {".yaml", ".yml"}
_CANDIDATE_NAMES = ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")


class YoloDatasetError(ValueError):
    """Raised when a dataset argument cannot be resolved to a YOLO ``data.yaml``."""


def resolve_dataset_yaml(dataset: str | Path) -> Path:
    """Resolve *dataset* to a concrete ``data.yaml`` path.

    Accepts either a path to a ``.yaml``/``.yml`` file, or a directory that
    contains ``data.yaml`` (or ``dataset.yaml``). Raises :class:`YoloDatasetError`
    if nothing usable is found.
    """
    p = Path(dataset)
    if p.is_dir():
        for cand in _CANDIDATE_NAMES:
            if (p / cand).is_file():
                return (p / cand).resolve()
        raise YoloDatasetError(f"no {' / '.join(_CANDIDATE_NAMES)} found in directory {p}")
    if p.suffix.lower() in _YAML_EXTS:
        if not p.is_file():
            raise YoloDatasetError(f"dataset YAML not found: {p}")
        return p.resolve()
    raise YoloDatasetError(
        f"unrecognized dataset {dataset!r}: expected a .yaml/.yml file or a "
        f"directory containing data.yaml"
    )


def _names_to_list(names: Any) -> list[str]:
    """Normalize a ``names`` field (list or ``{id: name}`` map) to an ordered list."""
    if isinstance(names, list):
        return [str(n) for n in names]
    if isinstance(names, dict):
        try:
            ordered = sorted(names.items(), key=lambda kv: int(kv[0]))
        except (ValueError, TypeError):
            ordered = list(names.items())
        return [str(v) for _, v in ordered]
    return []


def _count_images(split: Path | None) -> int:
    """Best-effort count of images referenced by a YOLO split path.

    A split may be a directory of images, or a ``.txt`` manifest listing image
    paths (one per line). Anything else counts as 0.
    """
    if split is None:
        return 0
    if split.is_dir():
        return sum(1 for f in split.rglob("*") if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)
    if split.is_file() and split.suffix.lower() == ".txt":
        try:
            return sum(1 for ln in split.read_text().splitlines() if ln.strip())
        except OSError:
            return 0
    return 0


def validate_yolo_yaml(dataset: str | Path) -> dict[str, Any]:
    """Validate a YOLO ``data.yaml`` against the training contract.

    Returns a structured verdict::

        {
          "status": "ok" | "failed",
          "dataset_format": "yolo",
          "yaml_path": str,
          "nc": int | None,
          "names": [...],
          "train": str | None,
          "val": str | None,
          "n_train_images": int,
          "uses_download_script": bool,   # present-but-NOT-executed
          "issues": [str, ...],
        }

    Checks performed: the YAML exists and parses to a mapping; ``names`` is
    present; ``nc`` is present or derivable from ``names``; ``nc`` matches
    ``len(names)``; ``train`` and ``val`` (or ``valid``) splits are declared and
    their resolved directories exist. Never executes a ``download:`` block.
    """
    issues: list[str] = []
    base_result: dict[str, Any] = {
        "status": "failed",
        "dataset_format": "yolo",
        "yaml_path": None,
        "nc": None,
        "names": [],
        "train": None,
        "val": None,
        "n_train_images": 0,
        "uses_download_script": False,
        "issues": issues,
    }

    try:
        yaml_path = resolve_dataset_yaml(dataset)
    except YoloDatasetError as exc:
        issues.append(str(exc))
        return base_result
    base_result["yaml_path"] = str(yaml_path)

    try:
        import yaml  # PyYAML (transitive dep; used by libreyolo too)
    except ImportError:  # pragma: no cover - PyYAML is effectively always present
        issues.append("PyYAML is required to parse data.yaml (pip install pyyaml)")
        return base_result

    try:
        cfg = yaml.safe_load(yaml_path.read_text()) or {}
    except Exception as exc:  # report any parse failure as a dataset issue
        issues.append(f"could not parse {yaml_path.name}: {exc}")
        return base_result
    if not isinstance(cfg, dict):
        issues.append("data.yaml must be a mapping (key: value pairs)")
        return base_result

    base_result["uses_download_script"] = bool(cfg.get("download"))

    names_list = _names_to_list(cfg.get("names"))
    base_result["names"] = names_list
    if not names_list:
        issues.append("missing 'names' (a list or an {id: name} mapping)")

    nc = cfg.get("nc")
    if nc is None and names_list:
        nc = len(names_list)
    if nc is None:
        issues.append("missing class count: provide 'nc' or 'names'")
    else:
        try:
            nc = int(nc)
            base_result["nc"] = nc
            if names_list and nc != len(names_list):
                issues.append(f"nc={nc} does not match len(names)={len(names_list)}")
        except (ValueError, TypeError):
            issues.append(f"'nc' must be an integer, got {cfg.get('nc')!r}")

    # Resolve the dataset root: an explicit 'path', else the yaml's directory.
    root = cfg.get("path")
    base_dir = Path(root) if root else yaml_path.parent
    if not base_dir.is_absolute():
        base_dir = (yaml_path.parent / base_dir).resolve()

    def _resolve_split(value: Any) -> Path | None:
        if not value or not isinstance(value, (str, Path)):
            return None
        sp = Path(value)
        return sp if sp.is_absolute() else (base_dir / sp)

    train_raw = cfg.get("train")
    val_raw = cfg.get("val") if cfg.get("val") is not None else cfg.get("valid")
    if not train_raw:
        issues.append("missing 'train' split")
    if not val_raw:
        issues.append("missing 'val' (or 'valid') split")

    train_p = _resolve_split(train_raw)
    val_p = _resolve_split(val_raw)
    base_result["train"] = str(train_p) if train_p else None
    base_result["val"] = str(val_p) if val_p else None

    if train_p is not None and not train_p.exists():
        issues.append(f"train path does not exist: {train_p}")
    if val_p is not None and not val_p.exists():
        issues.append(f"val path does not exist: {val_p}")

    base_result["n_train_images"] = _count_images(train_p)

    base_result["status"] = "ok" if not issues else "failed"
    return base_result


__all__ = ["YoloDatasetError", "resolve_dataset_yaml", "validate_yolo_yaml"]
