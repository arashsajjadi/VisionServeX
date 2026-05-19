# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.27.0: LibreYOLO engine integration.

LibreYOLO (https://github.com/Libre-YOLO/libreyolo, PyPI ``libreyolo`` MIT)
exposes unified YOLOX / YOLOv9 / YOLO-NAS / RT-DETR / RF-DETR / D-FINE
wrappers with a single inference API. v2.27 wires it into VisionServeX as
an optional engine.

Commands:
    visionservex libreyolo doctor        — checks libreyolo install + GPU
    visionservex libreyolo list-models   — discovers downloadable weights (HF + Deci CDN)
    visionservex libreyolo pull MODEL_ID — downloads a weight (license-gated)
    visionservex libreyolo smoke-test MODEL_ID IMG — runs inference; normalised output
    visionservex libreyolo license-audit — per-weight license verdict

Weight licenses (verified 2026-05-18):
- libreyolo-yolox-*  : YOLOX upstream Apache-2.0 (permissive). Pulled by default.
- libreyolo-yolo9-*  : YOLOv9 upstream GPL-3.0. Source code is GPL; weights
                       are derived. Marked LICENSE_RISK=gpl by default.
- libreyolo-yolonas-*: Deci.AI proprietary, non-commercial. **BLOCKED by
                       default** → ``LIBREYOLO_WEIGHT_LICENSE_NONCOMMERCIAL``.
- libreyolo-dfine-*  : D-FINE upstream Apache-2.0. Pulled by default.
- libreyolo-rtdetr-* : RT-DETR upstream Apache-2.0. Pulled by default.
- libreyolo-rfdetr-* : RF-DETR upstream Apache-2.0. Pulled by default.

Aggressive pull policy (v2.27 default): auto-pull permissive (Apache-2.0,
MIT) weights for benchmarking. GPL → opt-in. Non-commercial → blocked.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(
    help="v2.27.0: LibreYOLO (MIT engine) integration — doctor/discovery/pull/smoke/license.",
    no_args_is_help=True,
)
console = Console()

LIBREYOLO_REPO = "https://github.com/Libre-YOLO/libreyolo"
LIBREYOLO_MIN_VERSION = "1.1.0"

# Per-family weight license verdicts. Aggressive policy: pull permissive,
# gate non-commercial, warn-gate GPL.
WEIGHT_LICENSE_TABLE: dict[str, dict[str, Any]] = {
    "yolox": {
        "code_license": "MIT",
        "weight_license": "Apache-2.0",
        "weight_upstream": "https://github.com/Megvii-BaseDetection/YOLOX",
        "license_risk": "none",
        "auto_pull": True,
    },
    "yolo9": {
        "code_license": "MIT",
        "weight_license": "GPL-3.0",
        "weight_upstream": "https://github.com/WongKinYiu/yolov9",
        "license_risk": "gpl",
        "auto_pull": False,  # opt-in only
    },
    "yolonas": {
        "code_license": "MIT",
        "weight_license": "Deci-AI YOLO-NAS proprietary, non-commercial",
        "weight_upstream": "https://github.com/Deci-AI/super-gradients",
        "license_risk": "non_commercial",
        "auto_pull": False,  # blocked by default
    },
    "dfine": {
        "code_license": "MIT",
        "weight_license": "Apache-2.0",
        "weight_upstream": "https://github.com/Peterande/D-FINE",
        "license_risk": "none",
        "auto_pull": True,
    },
    "rtdetr": {
        "code_license": "MIT",
        "weight_license": "Apache-2.0",
        "weight_upstream": "https://github.com/lyuwenyu/RT-DETR",
        "license_risk": "none",
        "auto_pull": True,
    },
    "rfdetr": {
        "code_license": "MIT",
        "weight_license": "Apache-2.0",
        "weight_upstream": "https://github.com/roboflow/rf-detr",
        "license_risk": "none",
        "auto_pull": True,
    },
}


def _libreyolo_available() -> tuple[bool, str]:
    try:
        import libreyolo

        ver = getattr(libreyolo, "__version__", "unknown")
        return True, ver
    except ImportError:
        return False, ""


def _emit(payload: dict[str, Any], *, out: Path | None, fmt: str) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    color = {
        "ok": "green",
        "expected_blocker": "yellow",
        "failed": "red",
    }.get(payload.get("status", ""), "white")
    console.print(f"[{color}]{payload.get('code', '')}[/{color}]: {payload.get('message', '')}")


def _libreyolo_id_to_libre_class(model_id: str) -> tuple[str, str] | None:
    """Map ``libreyolo-<family>-<size>`` → (FAMILY, SIZE)."""
    if not model_id.startswith("libreyolo-"):
        return None
    parts = model_id[len("libreyolo-") :].split("-")
    if len(parts) < 2:
        return None
    family = parts[0]
    size = "-".join(parts[1:])  # rtdetr-r50m etc.
    return family, size


def _registry_classes() -> list:
    """Return the BaseModel registry with rfdetr lazy-loaded."""
    import libreyolo  # noqa: F401
    from libreyolo.models import _ensure_rfdetr
    from libreyolo.models.base.model import BaseModel

    try:
        _ensure_rfdetr()
    except Exception:
        pass
    return list(BaseModel._registry)


def _all_discovered_weights() -> list[dict[str, Any]]:
    """Probe libreyolo's BaseModel registry for downloadable weights."""
    weights: list[dict[str, Any]] = []
    for cls in _registry_classes():
        family = cls.FAMILY
        for size in cls.INPUT_SIZES:
            for task_suffix in ("", "-seg"):
                filename = f"{cls.FILENAME_PREFIX}{size}{task_suffix}{cls.WEIGHT_EXT}"
                url = cls.get_download_url(filename)
                if not url:
                    continue
                model_id = f"libreyolo-{family}-{size}" + ("-seg" if task_suffix else "")
                weights.append(
                    {
                        "model_id": model_id,
                        "family": family,
                        "size": size,
                        "task": "segment" if task_suffix else "detect",
                        "filename": filename,
                        "url": url,
                        "input_size": cls.INPUT_SIZES[size],
                    }
                )
    return weights


def _license_verdict_for_family(family: str) -> dict[str, Any]:
    return WEIGHT_LICENSE_TABLE.get(
        family,
        {
            "code_license": "MIT",
            "weight_license": "UNKNOWN",
            "weight_upstream": "",
            "license_risk": "unknown_weight_license",
            "auto_pull": False,
        },
    )


@app.command("doctor")
def doctor_cmd(
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Check that LibreYOLO is installed and report installed version + GPU."""
    avail, ver = _libreyolo_available()
    payload: dict[str, Any] = {
        "status": "ok" if avail else "expected_blocker",
        "code": "OK" if avail else "LIBREYOLO_REQUIRED",
        "libreyolo_installed": avail,
        "libreyolo_version": ver,
        "minimum_version": LIBREYOLO_MIN_VERSION,
        "upstream_repo": LIBREYOLO_REPO,
        "code_license": "MIT",
    }
    if avail:
        # GPU probe
        try:
            import torch

            payload["torch_version"] = torch.__version__
            payload["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                payload["gpu_name"] = torch.cuda.get_device_name(0)
                payload["compute_capability"] = list(torch.cuda.get_device_capability(0))
        except Exception:
            pass
        payload["message"] = f"libreyolo {ver} installed; GPU probe attempted."
    else:
        payload["message"] = "libreyolo not installed. Install via `pip install libreyolo` (MIT)."
    _emit(payload, out=out, fmt=fmt)


@app.command("list-models")
def list_models_cmd(
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
    head_check: bool = typer.Option(
        False,
        "--head-check/--no-head-check",
        help="HEAD each URL to confirm availability (slow; default off).",
    ),
) -> None:
    """Discover downloadable LibreYOLO weights from the BaseModel registry."""
    avail, ver = _libreyolo_available()
    if not avail:
        _emit(
            {
                "status": "expected_blocker",
                "code": "LIBREYOLO_REQUIRED",
                "message": "libreyolo not installed.",
            },
            out=out,
            fmt=fmt,
        )
        return

    weights = _all_discovered_weights()
    if head_check:
        import requests

        confirmed: list[dict[str, Any]] = []
        for w in weights:
            try:
                r = requests.head(w["url"], timeout=10, allow_redirects=True)
                w["http_status"] = r.status_code
                w["available"] = r.status_code == 200
                if r.status_code == 200:
                    w["size_bytes"] = int(r.headers.get("content-length", 0))
                    confirmed.append(w)
            except Exception as exc:
                w["available"] = False
                w["error"] = str(exc)[:200]
        weights = confirmed

    # Apply license verdict per row
    for w in weights:
        verdict = _license_verdict_for_family(w["family"])
        w.update(verdict)

    by_family: dict[str, int] = {}
    by_license_risk: dict[str, int] = {}
    by_task: dict[str, int] = {}
    for w in weights:
        by_family[w["family"]] = by_family.get(w["family"], 0) + 1
        by_license_risk[w["license_risk"]] = by_license_risk.get(w["license_risk"], 0) + 1
        by_task[w["task"]] = by_task.get(w["task"], 0) + 1

    payload = {
        "status": "ok",
        "code": "OK",
        "libreyolo_version": ver,
        "n_weights": len(weights),
        "by_family": by_family,
        "by_license_risk": by_license_risk,
        "by_task": by_task,
        "weights": weights,
    }
    _emit(payload, out=out, fmt=fmt)


@app.command("license-audit")
def license_audit_cmd(
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Per-family license verdict for LibreYOLO weights."""
    rows = []
    for family, verdict in WEIGHT_LICENSE_TABLE.items():
        rows.append({"family": family, **verdict})
    payload = {
        "status": "ok",
        "code": "OK",
        "n_families": len(rows),
        "rows": rows,
        "policy": (
            "Aggressive default: pull permissive (Apache-2.0/MIT). Opt-in "
            "for GPL-3.0. Blocked for non-commercial (YOLO-NAS)."
        ),
    }
    _emit(payload, out=out, fmt=fmt)


@app.command("pull")
def pull_cmd(
    model_id: str = typer.Argument(...),
    accept_gpl: bool = typer.Option(
        False,
        "--accept-gpl",
        help="Acknowledge GPL-3.0 weight terms (e.g. YOLOv9).",
    ),
    accept_noncommercial: bool = typer.Option(
        False,
        "--accept-noncommercial",
        help="Acknowledge non-commercial terms (e.g. YOLO-NAS / Deci proprietary).",
    ),
    cache_root: Path = typer.Option(
        Path.home() / ".cache" / "visionservex" / "libreyolo",
        "--cache-root",
    ),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
    timeout_s: int = typer.Option(600, "--timeout-s"),
) -> None:
    """Download a LibreYOLO weight to ``~/.cache/visionservex/libreyolo``."""
    import requests

    avail, _ver = _libreyolo_available()
    if not avail:
        _emit(
            {"status": "expected_blocker", "code": "LIBREYOLO_REQUIRED"},
            out=out,
            fmt=fmt,
        )
        return

    parsed = _libreyolo_id_to_libre_class(model_id)
    if parsed is None:
        _emit(
            {
                "status": "expected_blocker",
                "code": "LIBREYOLO_MODEL_NOT_FOUND",
                "model_id": model_id,
                "message": (
                    f"Unknown LibreYOLO id {model_id!r}. Expected libreyolo-<family>-<size>[-seg]."
                ),
            },
            out=out,
            fmt=fmt,
        )
        return
    family, size_or_seg = parsed
    task = "segment" if size_or_seg.endswith("-seg") else "detect"
    size = size_or_seg.removesuffix("-seg")

    verdict = _license_verdict_for_family(family)
    if verdict["license_risk"] == "non_commercial" and not accept_noncommercial:
        _emit(
            {
                "status": "expected_blocker",
                "code": "LIBREYOLO_WEIGHT_LICENSE_NONCOMMERCIAL",
                "model_id": model_id,
                "family": family,
                "license": verdict["weight_license"],
                "message": (
                    f"{family} weights are non-commercial. Pass "
                    "--accept-noncommercial to acknowledge."
                ),
            },
            out=out,
            fmt=fmt,
        )
        return
    if verdict["license_risk"] == "gpl" and not accept_gpl:
        _emit(
            {
                "status": "expected_blocker",
                "code": "LIBREYOLO_WEIGHT_LICENSE_GPL",
                "model_id": model_id,
                "family": family,
                "license": verdict["weight_license"],
                "message": (
                    f"{family} weights are GPL-3.0. Pass --accept-gpl to "
                    "acknowledge (derived works must also be GPL)."
                ),
            },
            out=out,
            fmt=fmt,
        )
        return
    if verdict["license_risk"] == "unknown_weight_license":
        _emit(
            {
                "status": "expected_blocker",
                "code": "LIBREYOLO_WEIGHT_LICENSE_UNVERIFIED",
                "model_id": model_id,
                "family": family,
                "message": "Weight license unverified; refusing to auto-pull.",
            },
            out=out,
            fmt=fmt,
        )
        return

    # Resolve the URL via the LibreYOLO base registry.
    cls_for_family = None
    for c in _registry_classes():
        if family == c.FAMILY:
            cls_for_family = c
            break
    if cls_for_family is None:
        _emit(
            {
                "status": "expected_blocker",
                "code": "LIBREYOLO_MODEL_NOT_FOUND",
                "model_id": model_id,
                "message": f"No registry class for family={family!r}.",
            },
            out=out,
            fmt=fmt,
        )
        return

    suffix = "-seg" if task == "segment" else ""
    filename = f"{cls_for_family.FILENAME_PREFIX}{size}{suffix}{cls_for_family.WEIGHT_EXT}"
    url = cls_for_family.get_download_url(filename)
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / filename
    if target.exists() and target.stat().st_size > 0:
        _emit(
            {
                "status": "ok",
                "code": "OK",
                "model_id": model_id,
                "family": family,
                "url": url,
                "cache_path": str(target),
                "size_bytes": target.stat().st_size,
                "license": verdict["weight_license"],
                "message": f"Already cached at {target}.",
            },
            out=out,
            fmt=fmt,
        )
        return

    t0 = time.time()
    try:
        with requests.get(url, stream=True, timeout=timeout_s) as r:
            r.raise_for_status()
            with target.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)
    except Exception as exc:
        target.unlink(missing_ok=True)
        _emit(
            {
                "status": "failed",
                "code": "DOWNLOAD_FAILED",
                "model_id": model_id,
                "url": url,
                "error": str(exc)[:300],
            },
            out=out,
            fmt=fmt,
        )
        return
    runtime = time.time() - t0
    _emit(
        {
            "status": "ok",
            "code": "OK",
            "model_id": model_id,
            "family": family,
            "url": url,
            "cache_path": str(target),
            "size_bytes": target.stat().st_size,
            "license": verdict["weight_license"],
            "runtime_s": round(runtime, 2),
            "message": f"Downloaded {target.stat().st_size:_} bytes to {target}.",
        },
        out=out,
        fmt=fmt,
    )


@app.command("smoke-test")
def smoke_test_cmd(
    model_id: str = typer.Argument(...),
    image: Path = typer.Argument(...),
    device: str = typer.Option("cuda", "--device"),
    cache_root: Path = typer.Option(
        Path.home() / ".cache" / "visionservex" / "libreyolo",
        "--cache-root",
    ),
    score_threshold: float = typer.Option(0.25, "--score-threshold"),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
    draw: Path | None = typer.Option(None, "--draw"),
) -> None:
    """Run a single-image smoke test on a LibreYOLO model."""
    avail, ver = _libreyolo_available()
    if not avail:
        _emit(
            {"status": "expected_blocker", "code": "LIBREYOLO_REQUIRED"},
            out=out,
            fmt=fmt,
        )
        return
    if not image.exists():
        _emit(
            {"status": "failed", "code": "INPUT_NOT_FOUND", "image": str(image)},
            out=out,
            fmt=fmt,
        )
        raise typer.Exit(2)

    parsed = _libreyolo_id_to_libre_class(model_id)
    if parsed is None:
        _emit(
            {"status": "expected_blocker", "code": "LIBREYOLO_MODEL_NOT_FOUND"},
            out=out,
            fmt=fmt,
        )
        return
    family, size_or_seg = parsed
    size = size_or_seg.removesuffix("-seg")

    cls_for_family = None
    for c in _registry_classes():
        if family == c.FAMILY:
            cls_for_family = c
            break
    if cls_for_family is None:
        _emit(
            {"status": "expected_blocker", "code": "LIBREYOLO_MODEL_NOT_FOUND"},
            out=out,
            fmt=fmt,
        )
        return

    filename = f"{cls_for_family.FILENAME_PREFIX}{size}{cls_for_family.WEIGHT_EXT}"
    weight_path = cache_root / filename
    if not weight_path.exists():
        _emit(
            {
                "status": "expected_blocker",
                "code": "CHECKPOINT_REQUIRED",
                "model_id": model_id,
                "expected_path": str(weight_path),
                "message": (f"Weight not cached. Run `visionservex libreyolo pull {model_id}`."),
            },
            out=out,
            fmt=fmt,
        )
        return

    try:
        model = cls_for_family(model_path=str(weight_path), size=size, device=device)
        t0 = time.time()
        result = model(source=str(image), save=False)
        elapsed = time.time() - t0
    except Exception as exc:
        _emit(
            {
                "status": "failed",
                "code": "LIBREYOLO_INFERENCE_FAILED",
                "model_id": model_id,
                "image": str(image),
                "device": device,
                "error": str(exc)[:500],
            },
            out=out,
            fmt=fmt,
        )
        return

    # Normalize output to canonical rows.
    rows = []
    try:
        # LibreYOLO returns Results object with .boxes
        boxes = getattr(result, "boxes", None)
        if boxes is None and isinstance(result, list) and result:
            boxes = getattr(result[0], "boxes", None)
        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy().tolist()
            conf = boxes.conf.detach().cpu().numpy().tolist()
            cls_ = boxes.cls.detach().cpu().numpy().tolist()
            try:
                names = boxes.names  # type: ignore[attr-defined]
            except AttributeError:
                names = {}
            for box, score, c in zip(xyxy, conf, cls_, strict=False):
                if score < score_threshold:
                    continue
                cid = int(c)
                rows.append(
                    {
                        "xyxy": [float(v) for v in box],
                        "score": float(score),
                        "class_id": cid,
                        "category_id": None,
                        "class_name": names.get(cid, f"class_{cid}")
                        if isinstance(names, dict)
                        else f"class_{cid}",
                        "source_engine": "libreyolo",
                    }
                )
    except Exception as exc:
        _emit(
            {
                "status": "failed",
                "code": "NORMALIZER_OUTPUT_INVALID",
                "model_id": model_id,
                "error": str(exc)[:300],
            },
            out=out,
            fmt=fmt,
        )
        return

    payload = {
        "status": "ok",
        "code": "OK",
        "model_id": model_id,
        "family": family,
        "image": str(image),
        "device": device,
        "n_predictions": len(rows),
        "forward_seconds": round(elapsed, 4),
        "predictions": rows[:10],
        "source_engine": "libreyolo",
        "libreyolo_version": ver,
    }
    _emit(payload, out=out, fmt=fmt)


@app.command("build-model-map")
def build_model_map_cmd(
    hf_audit: Path | None = typer.Option(
        None,
        "--hf-audit",
        help="Path to reports/libreyolo_hf_full_audit_v230.json (optional context).",
    ),
    out: Path = typer.Option(..., "--out"),
    fmt: str = typer.Option("json", "--format"),
) -> None:
    """v2.30.0: build the canonical LibreYOLO model-map.

    Produces one row per default-safe weight with:
      hf_model_id, weight_filename, libreyolo_load_name,
      VisionServeX_model_id, task, smoke_command, benchmark_command,
      license, source_url.
    """
    avail, ver = _libreyolo_available()
    if not avail:
        _emit(
            {
                "status": "expected_blocker",
                "code": "LIBREYOLO_REQUIRED",
                "message": "libreyolo is not installed",
                "fix": "pip install 'visionservex[libreyolo]'",
            },
            out=out,
            fmt=fmt,
        )
        return

    weights = _all_discovered_weights()
    rows: list[dict[str, Any]] = []
    for w in weights:
        verdict = _license_verdict_for_family(w["family"])
        license_str = (verdict.get("weight_license") or "").upper()
        risk = (verdict.get("license_risk") or "").lower()
        default_safe = risk == "none" and any(
            ok in license_str for ok in ("APACHE-2.0", "APACHE 2.0", "MIT")
        )

        # Compose hf_model_id heuristically from upstream URL
        url = w.get("url", "")
        hf_model_id = ""
        if "huggingface.co/" in url:
            rest = url.split("huggingface.co/", 1)[1]
            parts = rest.split("/resolve/", 1)
            if parts:
                hf_model_id = parts[0]

        rows.append(
            {
                "hf_model_id": hf_model_id,
                "weight_filename": w.get("filename", ""),
                "libreyolo_load_name": w.get("filename", ""),
                "visionservex_model_id": w.get("model_id", ""),
                "family": w.get("family", ""),
                "size": w.get("size", ""),
                "task": w.get("task", ""),
                "input_size": w.get("input_size"),
                "license": verdict.get("weight_license", ""),
                "license_risk": verdict.get("license_risk", ""),
                "source_url": url,
                "code_license": verdict.get("code_license", "MIT"),
                "default_safe": default_safe,
                "smoke_command": (
                    f"visionservex libreyolo smoke-test {w.get('model_id', '')} "
                    f"tests/assets/smoke/coco_person_car.jpg --device cuda --format json "
                    f"--out reports/libreyolo_{w.get('model_id', '')}_smoke_v230.json"
                ),
                "benchmark_command": (
                    f"visionservex benchmark-detection --dataset coco:COCO400 "
                    f"--models {w.get('model_id', '')} --backend libreyolo --device cuda "
                    f"--format json --out reports/libreyolo_{w.get('model_id', '')}_bench_v230.json"
                ),
            }
        )

    by_default_safe = sum(1 for r in rows if r["default_safe"])
    payload = {
        "status": "ok",
        "code": "OK",
        "version": "v2.30.0",
        "libreyolo_version": ver,
        "n_weights": len(rows),
        "n_default_safe": by_default_safe,
        "n_blocked": len(rows) - by_default_safe,
        "hf_audit_input": str(hf_audit) if hf_audit else "",
        "rows": rows,
    }
    _emit(payload, out=out, fmt=fmt)


__all__ = ["app"]


@app.command("contract-test-all-default-safe")
def contract_test_all_default_safe_cmd(
    device: str = typer.Option("cuda", "--device"),
    out: Path = typer.Option(..., "--out"),
    csv: Path | None = typer.Option(None, "--csv"),
    draw_dir: Path | None = typer.Option(None, "--draw-dir"),
    fmt: str = typer.Option("json", "--format"),
) -> None:
    """v2.34.0: contract-test every default-safe LibreYOLO weight."""
    import subprocess as _sp
    import sys as _sys
    import time as _time

    avail, ver = _libreyolo_available()
    if not avail:
        _emit(
            {
                "status": "expected_blocker",
                "code": "LIBREYOLO_REQUIRED",
                "fix": "pip install 'visionservex[libreyolo]'",
            },
            out=out,
            fmt=fmt,
        )
        return

    weights = _all_discovered_weights()
    rows: list[dict[str, Any]] = []
    for w in weights:
        verdict = _license_verdict_for_family(w["family"])
        risk = (verdict.get("license_risk") or "").lower()
        wl = (verdict.get("weight_license") or "").upper()
        is_safe = risk == "none" and any(ok in wl for ok in ("APACHE-2.0", "APACHE 2.0", "MIT"))
        if not is_safe:
            rows.append(
                {
                    "model_id": w["model_id"],
                    "family": w["family"],
                    "task": w["task"],
                    "final_state": "license_blocked",
                    "blocker_code": "LIBREYOLO_WEIGHT_LICENSE_NOT_DEFAULT_SAFE",
                    "weight_license": verdict.get("weight_license", ""),
                    "license_risk": risk,
                    "fix": "opt-in explicitly if you accept the license",
                }
            )
            continue

        smoke_img_path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "tests/assets/smoke/coco_person_car.jpg"
        )
        if not smoke_img_path.exists():
            smoke_img_path = Path("/tmp/smoke_test.jpg")

        out_json = Path(f"/tmp/libreyolo_contract_{w['model_id'].replace('/', '_')}.json")
        cmd = [
            _sys.executable,
            "-m",
            "visionservex",
            "libreyolo",
            "smoke-test",
            w["model_id"],
            str(smoke_img_path),
            "--device",
            device,
            "--format",
            "json",
            "--out",
            str(out_json),
        ]
        if draw_dir:
            draw_dir.mkdir(parents=True, exist_ok=True)
            cmd += ["--draw", str(draw_dir / f"{w['model_id'].replace('/', '_')}.jpg")]

        t0 = _time.monotonic()
        try:
            proc = _sp.run(cmd, capture_output=True, text=True, timeout=120)
            rt = (_time.monotonic() - t0) * 1000.0
            payload: dict[str, Any] = {}
            try:
                payload = json.loads(out_json.read_text()) if out_json.exists() else {}
            except Exception:
                pass
            if not payload:
                import re

                m_obj = re.search(r"\{.*\}", proc.stdout, re.DOTALL)
                if m_obj:
                    try:
                        payload = json.loads(m_obj.group(0))
                    except Exception:
                        pass

            final_state = (
                "contract_passed"
                if payload.get("status") == "ok"
                else (
                    "expected_blocker"
                    if payload.get("status") == "expected_blocker"
                    else "download_failed_retryable"
                    if "download" in (payload.get("code", "") or "").lower()
                    else "package_bug"
                    if proc.returncode != 0
                    else "dependency_required"
                )
            )

            rows.append(
                {
                    "model_id": w["model_id"],
                    "family": w["family"],
                    "task": w["task"],
                    "final_state": final_state,
                    "blocker_code": payload.get("code", ""),
                    "n_predictions": payload.get("n_predictions", 0),
                    "runtime_ms": round(rt, 0),
                    "weight_license": verdict.get("weight_license", ""),
                    "fix": payload.get("install", ""),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model_id": w["model_id"],
                    "family": w["family"],
                    "task": w["task"],
                    "final_state": "package_bug",
                    "blocker_code": str(exc)[:100],
                    "n_predictions": 0,
                    "runtime_ms": 0,
                    "weight_license": verdict.get("weight_license", ""),
                    "fix": "",
                }
            )

    n_safe = sum(1 for r in rows if r["final_state"] == "contract_passed")
    n_blocked = sum(1 for r in rows if r["final_state"] in ("license_blocked",))
    payload_out = {
        "status": "ok",
        "code": "OK",
        "version": "v2.34.0",
        "libreyolo_version": ver,
        "n_total": len(rows),
        "n_contract_passed": n_safe,
        "n_license_blocked": n_blocked,
        "rows": rows,
    }
    _emit(payload_out, out=out, fmt=fmt)
    if csv is not None:
        import csv as _csv

        csv.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0].keys()) if rows else ["model_id"]
        with open(csv, "w", newline="") as fh:
            w2 = _csv.DictWriter(fh, fieldnames=fields)
            w2.writeheader()
            for r in rows:
                w2.writerow(r)
