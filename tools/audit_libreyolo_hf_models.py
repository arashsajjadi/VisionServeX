#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""VisionServeX v2.30.0 — LibreYOLO Hugging Face model audit.

Discovers every Hugging Face repo under the ``LibreYOLO`` org and writes
a per-repo audit row with family, task, weight files, license source,
auto_pull policy, default_safe verdict, and blocker_code.

Usage:
    python tools/audit_libreyolo_hf_models.py \\
        --author LibreYOLO \\
        --out-json reports/libreyolo_hf_full_audit_v230.json \\
        --out-csv reports/libreyolo_hf_full_audit_v230.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

# Heuristic family classifier from repo name
_FAMILY_HINTS: list[tuple[str, str]] = [
    ("yolox", "yolox"),
    ("yolov9", "yolov9"),
    ("yolo9", "yolov9"),
    ("yolonas", "yolonas"),
    ("yolo-nas", "yolonas"),
    ("rfdetr-seg", "rfdetr-seg"),
    ("rfdetr", "rfdetr"),
    ("rf-detr", "rfdetr"),
    ("dfine", "dfine"),
    ("d-fine", "dfine"),
    ("rtdetr", "rtdetr"),
    ("rt-detr", "rtdetr"),
    ("damo", "damoyolo"),
]


def _infer_family(name: str) -> str:
    low = name.lower()
    for hint, family in _FAMILY_HINTS:
        if hint in low:
            return family
    return "unknown"


def _infer_task(name: str, tags: list[str]) -> str:
    low = name.lower()
    if "seg" in low or "segment" in low:
        return "instance_segmentation"
    if "track" in low:
        return "tracking"
    if "detection" in low or "detect" in low:
        return "detection"
    for tag in tags or []:
        tag_l = tag.lower()
        if "segment" in tag_l:
            return "instance_segmentation"
        if "detection" in tag_l:
            return "detection"
    return "detection"


_WEIGHT_EXTENSIONS = (".pt", ".pth", ".safetensors", ".onnx", ".engine")


def _classify_files(siblings: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"weight_files": [], "license_files": [], "readme_files": [], "all_files": []}
    largest_weight: dict[str, Any] | None = None
    for s in siblings:
        rname = s.get("rfilename", "")
        size = s.get("size", 0) or 0
        entry = {"rfilename": rname, "size": size}
        out["all_files"].append(entry)
        low = rname.lower()
        if any(low.endswith(ext) for ext in _WEIGHT_EXTENSIONS):
            out["weight_files"].append(entry)
            if largest_weight is None or size > (largest_weight.get("size") or 0):
                largest_weight = entry
        if "license" in low or "notice" in low:
            out["license_files"].append(entry)
        if "readme" in low:
            out["readme_files"].append(entry)
    out["largest_weight_file"] = largest_weight
    return out


def _audit_license(
    card_data: dict[str, Any], info_license: str | None, files: dict[str, Any]
) -> dict[str, Any]:
    """Determine license + license_source."""
    lic = None
    source = "unknown"
    if isinstance(card_data, dict) and card_data.get("license"):
        lic = card_data["license"]
        source = "hf_card"
    elif info_license:
        lic = info_license
        source = "hf_card"
    elif files.get("license_files"):
        lic = "see_license_file"
        source = "LICENSE_file"

    if not lic:
        return {"license": None, "license_source": "unknown", "auto_pull_allowed": False}

    norm = str(lic).strip().lower()
    permissive = norm in {
        "mit",
        "apache-2.0",
        "apache2.0",
        "apache 2.0",
        "bsd-3-clause",
        "bsd-2-clause",
    }
    return {
        "license": str(lic),
        "license_source": source,
        "auto_pull_allowed": permissive,
    }


def _verdict_for_repo(
    family: str, task: str, files: dict[str, Any], lic_info: dict[str, Any]
) -> dict[str, Any]:
    """Compute final default_safe + blocker_code per row."""
    out = {
        "default_safe": False,
        "blocker_code": "",
    }

    if not files.get("weight_files"):
        out["blocker_code"] = "LIBREYOLO_NO_WEIGHT_FILE_FOUND"
        return out

    if not lic_info.get("license"):
        out["blocker_code"] = "LIBREYOLO_WEIGHT_LICENSE_UNVERIFIED"
        return out

    if not lic_info.get("auto_pull_allowed"):
        out["blocker_code"] = "LIBREYOLO_WEIGHT_LICENSE_NOT_DEFAULT_SAFE"
        return out

    if task == "unknown":
        out["blocker_code"] = "LIBREYOLO_UNSUPPORTED_TASK"
        return out

    if family in ("yolonas",):
        # Special-case: even if HF says permissive, YOLO-NAS upstream license is non-commercial.
        out["blocker_code"] = "LIBREYOLO_WEIGHT_LICENSE_NOT_DEFAULT_SAFE"
        return out

    out["default_safe"] = True
    return out


def audit(author: str, *, out_json: Path, out_csv: Path | None = None) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi, model_info
    except ImportError:
        result = {
            "status": "expected_blocker",
            "code": "HUGGINGFACE_HUB_REQUIRED",
            "message": "huggingface_hub not installed",
            "fix": "pip install huggingface_hub",
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2))
        return result

    api = HfApi()
    try:
        models = list(api.list_models(author=author, full=True))
    except Exception as exc:
        result = {
            "status": "expected_blocker",
            "code": "HF_NETWORK_FAILED",
            "message": f"Could not list models for author={author!r}: {exc}",
            "fix": "Check network / HF_HUB token; retry.",
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2))
        return result

    rows: list[dict[str, Any]] = []
    for m in models:
        try:
            info = model_info(m.modelId, files_metadata=True)
        except Exception as exc:
            rows.append(
                {
                    "hf_model_id": m.modelId,
                    "error": f"model_info failed: {exc}"[:200],
                    "default_safe": False,
                    "blocker_code": "LIBREYOLO_MODEL_CARD_MISSING",
                }
            )
            continue

        siblings = []
        for s in info.siblings or []:
            siblings.append(
                {
                    "rfilename": getattr(s, "rfilename", ""),
                    "size": getattr(s, "size", None),
                    "blob_id": getattr(s, "blob_id", None),
                }
            )
        card_data = info.cardData or {}
        tags = list(getattr(info, "tags", []) or [])

        family = _infer_family(m.modelId)
        task = _infer_task(m.modelId, tags)
        files = _classify_files(siblings)
        lic_info = _audit_license(card_data, getattr(info, "license", None), files)
        verdict = _verdict_for_repo(family, task, files, lic_info)

        largest = files.get("largest_weight_file") or {}
        row = {
            "hf_model_id": m.modelId,
            "display_name": str(card_data.get("display_name", m.modelId)),
            "family": family,
            "task": task,
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "library_name": card_data.get("library_name"),
            "tags": tags,
            "n_weight_files": len(files["weight_files"]),
            "weight_filename": largest.get("rfilename", ""),
            "weight_size_bytes": largest.get("size"),
            "n_license_files": len(files["license_files"]),
            "n_readme_files": len(files["readme_files"]),
            "license": lic_info["license"],
            "license_source": lic_info["license_source"],
            "auto_pull_allowed": lic_info["auto_pull_allowed"],
            "default_safe": verdict["default_safe"],
            "blocker_code": verdict["blocker_code"],
            "downloads": getattr(info, "downloads", None),
            "last_modified": str(getattr(info, "lastModified", "")),
        }
        rows.append(row)

    n_default_safe = sum(1 for r in rows if r.get("default_safe"))
    summary = {
        "version": "v2.30.0",
        "author": author,
        "n_models": len(rows),
        "n_default_safe": n_default_safe,
        "n_license_blocked": sum(
            1 for r in rows if r.get("blocker_code") == "LIBREYOLO_WEIGHT_LICENSE_NOT_DEFAULT_SAFE"
        ),
        "n_license_unverified": sum(
            1 for r in rows if r.get("blocker_code") == "LIBREYOLO_WEIGHT_LICENSE_UNVERIFIED"
        ),
        "n_no_weights": sum(
            1 for r in rows if r.get("blocker_code") == "LIBREYOLO_NO_WEIGHT_FILE_FOUND"
        ),
        "rows": rows,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, default=str))

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            fields = [
                "hf_model_id",
                "family",
                "task",
                "weight_filename",
                "weight_size_bytes",
                "license",
                "license_source",
                "auto_pull_allowed",
                "default_safe",
                "blocker_code",
                "downloads",
                "last_modified",
            ]
            with open(out_csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow(r)

    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--author", default="LibreYOLO")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args()

    summary = audit(args.author, out_json=args.out_json, out_csv=args.out_csv)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, default=str))


if __name__ == "__main__":
    main()
