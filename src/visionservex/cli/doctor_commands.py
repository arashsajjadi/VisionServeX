# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.33.0: extra-by-extra doctor commands.

Each doctor reports the exact status of an optional extra so users get an
actionable install command for any missing dependency.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(
    help="v2.33.0: doctor commands per optional extra.",
    no_args_is_help=True,
)


def _check(
    modules: list[str], install_cmd: str, name: str, *, conflict_check: dict | None = None
) -> dict[str, Any]:
    """Probe a list of import names; return doctor report."""
    installed = {}
    missing = []
    for m in modules:
        try:
            mod = importlib.import_module(m)
            installed[m] = getattr(mod, "__version__", "imported_ok")
        except Exception as exc:
            missing.append({"module": m, "error": str(exc)[:200]})

    ok = not missing
    out = {
        "status": "ok" if ok else "expected_blocker",
        "code": "OK" if ok else f"{name.upper()}_REQUIRED",
        "extra": name,
        "installed": installed,
        "missing": [m["module"] for m in missing],
        "missing_detail": missing,
        "exact_install_command": install_cmd,
        "sidecar_recommended": False,
        "next_action": f"pip install '{install_cmd}'" if not ok else "use the feature",
    }

    # Conflict check
    if conflict_check and ok:
        for mod_name, (min_ver, max_ver) in conflict_check.items():
            v = installed.get(mod_name, "")
            if v and v != "imported_ok":
                from packaging.version import Version

                try:
                    ver = Version(v)
                    if min_ver and ver < Version(min_ver):
                        out["status"] = "expected_blocker"
                        out["code"] = f"{name.upper()}_VERSION_TOO_OLD"
                        out["next_action"] = f"pip install '{mod_name}>={min_ver}'"
                    if max_ver and ver >= Version(max_ver):
                        out["status"] = "expected_blocker"
                        out["code"] = f"{name.upper()}_VERSION_TOO_NEW"
                        out["next_action"] = (
                            f"pip install '{mod_name}<{max_ver}' (sidecar recommended)"
                        )
                        out["sidecar_recommended"] = True
                except Exception:
                    pass

    return out


def _emit(report: dict, out: Path | None, fmt: str) -> None:
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
    typer.echo(json.dumps(report, indent=2))


@app.command("all-benchmark")
def doctor_all_benchmark(
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("json", "--format"),
) -> None:
    """Audit the all-benchmark extra (notebook + benchmark + detection + segmentation)."""
    modules = [
        "torch",
        "torchvision",
        "pycocotools",
        "matplotlib",
        "pandas",
        "numpy",
        "PIL",
        "tqdm",
        "transformers",
        "rfdetr",
    ]
    report = _check(modules, "visionservex[all-benchmark]", "all_benchmark")
    _emit(report, out, fmt)


@app.command("detection")
def doctor_detection(
    out: Path | None = typer.Option(None, "--out"), fmt: str = typer.Option("json", "--format")
) -> None:
    """Audit the detection extra."""
    modules = ["rfdetr", "supervision", "transformers", "timm", "huggingface_hub"]
    report = _check(modules, "visionservex[detection]", "detection")
    _emit(report, out, fmt)


@app.command("segmentation")
def doctor_segmentation(
    out: Path | None = typer.Option(None, "--out"), fmt: str = typer.Option("json", "--format")
) -> None:
    """Audit the segmentation extra."""
    modules = ["pycocotools", "rfdetr", "supervision", "cv2"]
    report = _check(modules, "visionservex[segmentation]", "segmentation")
    _emit(report, out, fmt)


@app.command("promptable")
def doctor_promptable(
    out: Path | None = typer.Option(None, "--out"), fmt: str = typer.Option("json", "--format")
) -> None:
    """Audit the promptable extra (SAM/SAM2)."""
    modules = ["transformers", "huggingface_hub", "pycocotools"]
    report = _check(modules, "visionservex[promptable]", "promptable")
    _emit(report, out, fmt)


@app.command("foundation")
def doctor_foundation(
    out: Path | None = typer.Option(None, "--out"), fmt: str = typer.Option("json", "--format")
) -> None:
    """Audit the foundation extra (timm + transformers)."""
    modules = ["timm", "transformers", "huggingface_hub", "safetensors"]
    report = _check(modules, "visionservex[foundation]", "foundation")
    _emit(report, out, fmt)


@app.command("dino")
def doctor_dino(
    out: Path | None = typer.Option(None, "--out"), fmt: str = typer.Option("json", "--format")
) -> None:
    """Audit the dino extra (DINOv3 — needs transformers>=4.56)."""
    modules = ["transformers", "timm", "huggingface_hub", "safetensors"]
    report = _check(
        modules, "visionservex[dino]", "dino", conflict_check={"transformers": ("4.56", None)}
    )
    _emit(report, out, fmt)


@app.command("sam3")
def doctor_sam3(
    out: Path | None = typer.Option(None, "--out"), fmt: str = typer.Option("json", "--format")
) -> None:
    """Audit SAM3 / SAM3.1 (Meta promptable segmentation/tracking)."""
    report = _check(["transformers", "huggingface_hub"], "visionservex[promptable]", "sam3")
    report["official_repo"] = "https://github.com/facebookresearch/segment-anything-3"
    report["hf_org"] = "facebook"
    report["models"] = ["facebook/sam3", "facebook/sam3.1-base", "facebook/sam3.1-large"]
    report["auth_required"] = True
    report["auth_instructions"] = (
        "1. Visit https://huggingface.co/facebook/sam3.\n"
        "2. Accept the model license.\n"
        "3. Set HF_TOKEN environment variable.\n"
        "4. Run: huggingface-cli login"
    )
    report["status"] = "auth_required"
    report["code"] = "SAM3_AUTH_REQUIRED"
    report["next_action"] = "Authenticate with Hugging Face and accept the SAM3 license."
    _emit(report, out, fmt)


@app.command("grounding-dino15")
def doctor_grounding_dino15(
    out: Path | None = typer.Option(None, "--out"), fmt: str = typer.Option("json", "--format")
) -> None:
    """Audit Grounding DINO 1.5/1.6 (IDEA-Research/DINO-X-API)."""
    report = _check(
        ["transformers", "huggingface_hub"], "visionservex[open-vocab]", "grounding_dino15"
    )
    report["official_repo"] = "https://github.com/IDEA-Research/Grounding-DINO-1.5-API"
    report["api_required"] = True
    report["api_key_env_var"] = "DINO_X_API_KEY"
    report["status"] = "auth_required"
    report["code"] = "GROUNDING_DINO15_API_KEY_REQUIRED"
    report["next_action"] = (
        "Request an API key from IDEA-Research and export it as "
        "DINO_X_API_KEY before using Grounding DINO 1.5/1.6."
    )
    _emit(report, out, fmt)


@app.command("florence2")
def doctor_florence2(
    out: Path | None = typer.Option(None, "--out"), fmt: str = typer.Option("json", "--format")
) -> None:
    """Audit Florence-2 (requires transformers >=4.40, <5.0)."""
    report = _check(
        ["transformers", "huggingface_hub", "safetensors", "accelerate"],
        "visionservex[vlm-legacy]",
        "florence2",
        conflict_check={"transformers": ("4.40", "5.0")},
    )
    if report.get("code") in ("FLORENCE2_VERSION_TOO_NEW",):
        report["code"] = "FLORENCE2_TRANSFORMERS_VERSION_REQUIRED"
        report["sidecar_recommended"] = True
        report["sidecar_install"] = (
            "python3.11 -m venv .venv-florence2 && source .venv-florence2/bin/activate "
            "&& pip install 'transformers>=4.40,<5.0' accelerate safetensors"
        )
    _emit(report, out, fmt)


@app.command("tracking")
def doctor_tracking(
    out: Path | None = typer.Option(None, "--out"), fmt: str = typer.Option("json", "--format")
) -> None:
    """Audit tracking extras (bytetracker, ocsort, torchreid)."""
    modules_to_check = ["bytetracker", "ocsort", "torchreid"]
    installed: dict[str, str] = {}
    missing: list[dict] = []
    for m in modules_to_check:
        try:
            mod = importlib.import_module(m)
            installed[m] = getattr(mod, "__version__", "imported_ok")
        except Exception as exc:
            missing.append({"module": m, "error": str(exc)[:150]})

    report = {
        "status": "ok" if not missing else "dependency_required",
        "code": "OK" if not missing else "TRACKING_DEPS_REQUIRED",
        "extra": "tracking",
        "installed": installed,
        "missing": [m["module"] for m in missing],
        "missing_detail": missing,
        "install_commands": {
            "bytetracker": "pip install bytetracker",
            "ocsort": "pip install ocsort",
            "torchreid": "pip install torchreid (or git+https://github.com/KaiyangZhou/deep-person-reid)",
        },
        "next_action": "Install the specific tracking package you need.",
    }
    _emit(report, out, fmt)


@app.command("anomaly")
def doctor_anomaly(
    out: Path | None = typer.Option(None, "--out"), fmt: str = typer.Option("json", "--format")
) -> None:
    """Audit anomaly extra (anomalib)."""
    report = _check(["anomalib"], "visionservex[anomaly]", "anomaly")
    _emit(report, out, fmt)


@app.command("openmmlab")
def doctor_openmmlab(
    out: Path | None = typer.Option(None, "--out"), fmt: str = typer.Option("json", "--format")
) -> None:
    """Audit OpenMMLab (sidecar recommended due to Python/CUDA compatibility)."""
    report = _check(["mmcv", "mmdet"], "openmim install mmcv mmdet", "openmmlab")
    if report["status"] != "ok":
        report["sidecar_recommended"] = True
        report["sidecar_install"] = "visionservex sidecar create openmmlab"
        report["code"] = "OPENMMLAB_REQUIRED"
    _emit(report, out, fmt)


__all__ = ["app"]
