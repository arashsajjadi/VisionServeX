# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.22.0: DEIMv2 doctor / pull / smoke-test CLI.

Honest integration attempt against the real upstream state:

Upstream truth (verified 2026-05-18):
- Repo:        https://github.com/Intellindust-AI-Lab/DEIMv2 (Apache-2.0)
- HF cards:    Intellindust/DEIMv2_DINOv3_S_COCO uses
               huggingface_hub.PyTorchModelHubMixin (config.json +
               model.safetensors). NO HF Transformers model_type, so
               `transformers.AutoModelForObjectDetection.from_pretrained`
               cannot load it.
- PyPI:        ``deimv2`` is NOT on PyPI. Inference requires `git clone`
               of the upstream repo and adding it to PYTHONPATH.
- Requirements: torch==2.5.1 (STRICT PIN) per upstream requirements.txt.
               The current VisionServeX install on RTX 5080 ships
               torch 2.11.0+cu130 → STRICT VERSION CONFLICT.

The CLI surfaces below do REAL diagnostics:
- ``deimv2 doctor`` probes the upstream-package import, the torch version,
  and the HF checkpoint reachability.
- ``deimv2 pull`` attempts the HF download via ``huggingface_hub`` (the
  only path that doesn't require the upstream repo). Reports exact blocker
  if the upstream Python class isn't available.
- ``deimv2 smoke-test`` returns ``status=expected_blocker`` with the exact
  combined blocker (``TORCH_VERSION_CONFLICT`` + ``NEEDS_UPSTREAM_REPO``)
  unless both are resolved.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(
    help="v2.22.0: DEIMv2 (Real-Time Object Detection Meets DINOv3) doctor / pull / smoke-test.",
    no_args_is_help=True,
)
console = Console()

DEIMV2_UPSTREAM_REPO = "https://github.com/Intellindust-AI-Lab/DEIMv2"
DEIMV2_REQUIRED_TORCH = "2.5.1"

# Known DEIMv2 HuggingFace checkpoint card → upstream-class hint.
DEIMV2_HF_CHECKPOINTS: dict[str, dict[str, str]] = {
    "deimv2-s": {
        "hf_repo": "Intellindust/DEIMv2_DINOv3_S_COCO",
        "upstream_class": "deimv2.DEIMv2",
        "notes": "Small variant (research target ~50.9 AP, ~5.78 ms latency).",
    },
    "deimv2-m": {"hf_repo": "", "upstream_class": "", "notes": "HF card not yet published."},
    "deimv2-l": {"hf_repo": "", "upstream_class": "", "notes": "HF card not yet published."},
    "deimv2-x": {
        "hf_repo": "",
        "upstream_class": "",
        "notes": (
            "X variant — highest-priority accuracy candidate "
            "(~57.8 AP / ~13.75 ms). HF card not yet published."
        ),
    },
}


def _diagnose_environment() -> dict[str, Any]:
    """Return a structured snapshot of the local environment vs DEIMv2 needs."""
    info: dict[str, Any] = {
        "required_torch": DEIMV2_REQUIRED_TORCH,
        "installed_torch": None,
        "torch_version_match": False,
        "deimv2_package_importable": False,
        "huggingface_hub_available": False,
    }
    try:
        import torch  # type: ignore

        info["installed_torch"] = torch.__version__
        # The upstream pin is `==2.5.1`; we accept patch-level compatibility
        # only at the 2.5.x line because the project explicitly pinned it.
        installed = torch.__version__.split("+")[0]
        info["torch_version_match"] = installed.startswith("2.5")
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = repr(exc)[:200]
    try:
        importlib.import_module("deimv2")
        info["deimv2_package_importable"] = True
    except Exception:
        info["deimv2_package_importable"] = False
    try:
        importlib.import_module("huggingface_hub")
        info["huggingface_hub_available"] = True
    except Exception:
        info["huggingface_hub_available"] = False
    return info


def _emit(payload: dict[str, Any], *, out: Path | None, fmt: str) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2))
    else:
        color = {
            "ok": "green",
            "expected_blocker": "yellow",
            "failed": "red",
        }.get(payload.get("status", ""), "white")
        console.print(f"[{color}]{payload.get('code', '')}[/{color}]: {payload.get('message', '')}")


@app.command("create-env")
def create_env_cmd(
    dry_run: bool = typer.Option(True, "--dry-run/--execute", help="Default: plan only."),
    timeout_s: int = typer.Option(
        1800,
        "--timeout-s",
        help="Per-command timeout (default 1800s = 30 min).",
    ),
    profile: str = typer.Option(
        "",
        "--profile",
        help=(
            "Runtime profile: deimv2-cu124-stable (default), deimv2-blackwell-nightly, "
            "deimv2-cpu-proof, deimv2-a100-l4-fallback. Blackwell-nightly uses the "
            "PyTorch cu128 nightly index so RTX 5080 sm_120 is supported."
        ),
    ),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """v2.23.0+: plan (default) or execute creation of the DEIMv2 sidecar conda env."""
    from visionservex.sidecars import SidecarConfig, SidecarManager

    cfg = SidecarConfig(timeout_s=timeout_s)
    payload = SidecarManager().create(
        "deimv2", dry_run=dry_run, config=cfg, profile=(profile or None)
    )
    _emit(payload, out=out, fmt=fmt)


@app.command("doctor")
def doctor_cmd(
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Probe torch / deimv2 / huggingface_hub for DEIMv2 readiness."""
    env = _diagnose_environment()
    blockers = []
    if not env["torch_version_match"]:
        blockers.append("TORCH_VERSION_CONFLICT")
    if not env["deimv2_package_importable"]:
        blockers.append("NEEDS_UPSTREAM_REPO")
    if not env["huggingface_hub_available"]:
        blockers.append("HUGGINGFACE_HUB_REQUIRED")

    if not blockers:
        payload = {
            "status": "ok",
            "code": "OK",
            "message": "DEIMv2 environment looks complete.",
            "details": env,
            "upstream_repo": DEIMV2_UPSTREAM_REPO,
        }
    else:
        payload = {
            "status": "expected_blocker",
            "code": "DEIMV2_NOT_RUNNABLE",
            "blockers": blockers,
            "message": (
                "DEIMv2 cannot run in this environment. Combined blockers: "
                + ", ".join(blockers)
                + "."
            ),
            "upstream_repo": DEIMV2_UPSTREAM_REPO,
            "remediation": [
                f"git clone {DEIMV2_UPSTREAM_REPO}",
                "cd DEIMv2 && pip install -r requirements.txt",
                (
                    f"Note: upstream pins torch=={DEIMV2_REQUIRED_TORCH}; "
                    f"installed is {env.get('installed_torch')}. "
                    "Use a separate conda/venv environment to avoid breaking VisionServeX."
                ),
                "Then add DEIMv2 to PYTHONPATH and rerun `visionservex deimv2 smoke-test`.",
            ],
            "details": env,
        }
    _emit(payload, out=out, fmt=fmt)


@app.command("pull")
def pull_cmd(
    model_id: str = typer.Argument("deimv2-s"),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Download the DEIMv2 HF checkpoint (does not require the upstream repo).

    The checkpoint download alone is not enough to run DEIMv2 — the upstream
    Python class is required to instantiate the model. ``pull`` is provided
    for environments that already have the upstream repo on PYTHONPATH.
    """
    info = DEIMV2_HF_CHECKPOINTS.get(model_id)
    if info is None or not info.get("hf_repo"):
        payload = {
            "status": "expected_blocker",
            "code": "CHECKPOINT_NOT_FOUND",
            "model_id": model_id,
            "message": (
                f"No published HF checkpoint for {model_id!r} as of 2026-05-18. "
                "Only deimv2-s is publicly available on HuggingFace."
            ),
            "known_checkpoints": {k: v["hf_repo"] for k, v in DEIMV2_HF_CHECKPOINTS.items()},
            "upstream_repo": DEIMV2_UPSTREAM_REPO,
        }
        _emit(payload, out=out, fmt=fmt)
        return

    env = _diagnose_environment()
    if not env["huggingface_hub_available"]:
        _emit(
            {
                "status": "expected_blocker",
                "code": "HUGGINGFACE_HUB_REQUIRED",
                "model_id": model_id,
                "message": "Install `huggingface_hub` first: pip install huggingface_hub.",
                "details": env,
            },
            out=out,
            fmt=fmt,
        )
        return

    try:
        from huggingface_hub import snapshot_download  # type: ignore

        local = snapshot_download(repo_id=info["hf_repo"])
        payload = {
            "status": "ok",
            "code": "OK",
            "model_id": model_id,
            "hf_repo": info["hf_repo"],
            "cache_path": str(local),
            "message": (
                "HF snapshot downloaded. NOTE: running inference still requires the upstream "
                f"DEIMv2 Python class ({info['upstream_class']!r}). See `deimv2 doctor`."
            ),
            "upstream_repo": DEIMV2_UPSTREAM_REPO,
        }
    except Exception as exc:
        payload = {
            "status": "failed",
            "code": "HF_DOWNLOAD_FAILED",
            "model_id": model_id,
            "hf_repo": info["hf_repo"],
            "message": f"HF download failed: {exc!s:.300}",
        }
    _emit(payload, out=out, fmt=fmt)


@app.command("smoke-test")
def smoke_test_cmd(
    model_id: str = typer.Argument("deimv2-s"),
    image: Path = typer.Argument(..., help="Image path."),
    device: str = typer.Option("cuda", "--device"),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
    draw: Path | None = typer.Option(None, "--draw"),
) -> None:
    """Attempt DEIMv2 inference. Returns the exact structured blocker if it can't run."""
    if not image.exists():
        _emit(
            {
                "status": "failed",
                "code": "INPUT_NOT_FOUND",
                "image": str(image),
                "message": f"Image not found: {image}",
            },
            out=out,
            fmt=fmt,
        )
        raise typer.Exit(2)

    env = _diagnose_environment()
    if not env["torch_version_match"] or not env["deimv2_package_importable"]:
        blockers = []
        if not env["torch_version_match"]:
            blockers.append("TORCH_VERSION_CONFLICT")
        if not env["deimv2_package_importable"]:
            blockers.append("NEEDS_UPSTREAM_REPO")
        payload = {
            "status": "expected_blocker",
            "code": "DEIMV2_NOT_RUNNABLE",
            "model_id": model_id,
            "image": str(image),
            "device": device,
            "blockers": blockers,
            "message": (
                "DEIMv2 smoke-test cannot run because the upstream Python class is not "
                "importable and/or torch version does not match. Run `visionservex deimv2 "
                "doctor` for remediation."
            ),
            "details": env,
            "upstream_repo": DEIMV2_UPSTREAM_REPO,
        }
        _emit(payload, out=out, fmt=fmt)
        # Honest exit-0: this is an expected blocker, not a failure.
        return

    # If we reach here, environment looks complete. Attempt actual inference.
    info = DEIMV2_HF_CHECKPOINTS.get(model_id) or {"hf_repo": "", "upstream_class": "deimv2.DEIMv2"}
    try:
        import deimv2  # type: ignore

        cls = getattr(deimv2, "DEIMv2", None)
        if cls is None:
            raise ImportError("deimv2 module loaded but `DEIMv2` class not found.")
        if not info.get("hf_repo"):
            raise RuntimeError(f"No HF checkpoint published for {model_id!r}.")
        model = cls.from_pretrained(info["hf_repo"])
        model.eval()
        # Minimal smoke: load the image and call predict-style API. Without the
        # exact upstream API we can only attempt a best-effort dispatch.
        from PIL import Image as _PIL

        img = _PIL.open(image).convert("RGB")
        # Best-effort call — the upstream may expose .predict(img) or model(img).
        try:
            result = model.predict(img)  # type: ignore[attr-defined]
        except AttributeError:
            result = model(img)
        n = len(result) if hasattr(result, "__len__") else 0
        payload = {
            "status": "ok",
            "code": "OK",
            "model_id": model_id,
            "image": str(image),
            "device": device,
            "n_detections": n,
            "message": "DEIMv2 smoke-test loaded the model and produced a result object.",
        }
    except Exception as exc:
        payload = {
            "status": "failed",
            "code": "DEIMV2_LOADER_FAILED",
            "model_id": model_id,
            "image": str(image),
            "device": device,
            "error": str(exc)[:500],
            "message": (
                "DEIMv2 environment passed doctor but the upstream loader still failed. "
                "Check the exact `deimv2` API (it changes between commits)."
            ),
        }
    _emit(payload, out=out, fmt=fmt)


@app.command("benchmark")
def benchmark_cmd(
    image_dir: Path = typer.Argument(
        ...,
        help="Directory of images to run inference over.",
    ),
    model_id: str = typer.Option("deimv2-s", "--model-id"),
    profile: str = typer.Option(
        "deimv2-blackwell-nightly",
        "--profile",
        help="Sidecar profile (deimv2-blackwell-nightly = RTX 5080 sm_120).",
    ),
    max_images: int = typer.Option(20, "--max-images"),
    score_threshold: float = typer.Option(0.25, "--score-threshold"),
    device: str = typer.Option("cuda", "--device"),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Where to write the summary JSON.",
    ),
    out_ndjson: Path | None = typer.Option(
        None,
        "--out-ndjson",
        help="Where to write per-image NDJSON predictions (default: alongside --out).",
    ),
    fmt: str = typer.Option("text", "--format"),
    timeout_s: int = typer.Option(900, "--timeout-s"),
) -> None:
    """v2.26.0: DEIMv2 GPU latency probe + canonical detections.

    Runs DEIMv2 inside the sidecar env (default profile
    ``deimv2-blackwell-nightly``) on every image in ``IMAGE_DIR`` up to
    ``--max-images``. Records per-image forward latency, p50/p95, total
    predictions, and writes per-image predictions to NDJSON.

    This is a **latency + predictions probe**, not a scientific AP
    benchmark. AP requires a matching COCO annotation file (use
    ``visionservex dataset prepare-coco-val2017-subset`` + a separate
    AP-eval CLI).
    """
    from visionservex.sidecars import SidecarConfig, SidecarManager
    from visionservex.sidecars.manager import _resolve_sidecar_root

    if not image_dir.exists():
        _emit(
            {
                "status": "failed",
                "code": "INPUT_NOT_FOUND",
                "message": f"--image-dir {image_dir} not found.",
            },
            out=out,
            fmt=fmt,
        )
        raise typer.Exit(2)

    mgr = SidecarManager()
    if not mgr.env_exists("deimv2", profile=profile):
        _emit(
            {
                "status": "expected_blocker",
                "code": "SIDECAR_ENV_MISSING",
                "model_id": model_id,
                "profile": profile,
                "image_dir": str(image_dir),
                "message": (
                    f"Sidecar env for profile {profile!r} not installed. Run "
                    f"`visionservex deimv2 create-env --execute --profile {profile}`."
                ),
            },
            out=out,
            fmt=fmt,
        )
        return

    repo_root = _resolve_sidecar_root() / "deimv2"
    if not repo_root.exists():
        _emit(
            {
                "status": "expected_blocker",
                "code": "UPSTREAM_REPO_NOT_FOUND",
                "expected": str(repo_root),
                "message": "DEIMv2 upstream repo not cloned. Run create-env --execute.",
            },
            out=out,
            fmt=fmt,
        )
        return

    if out_ndjson is None:
        out_ndjson = out.with_suffix(".ndjson")

    env_name = mgr.env_name("deimv2", profile=profile)
    import subprocess as _sp
    import sys as _sys

    runner_path = (
        Path(_sys.modules["visionservex"].__file__).parent
        / "sidecars"
        / "_deimv2_benchmark_runner.py"
    )
    cmd = [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        str(runner_path),
        "--repo-root",
        str(repo_root),
        "--model-id",
        model_id,
        "--hf-repo",
        "Intellindust/DEIMv2_DINOv3_S_COCO",
        "--image-dir",
        str(image_dir),
        "--max-images",
        str(max_images),
        "--score-threshold",
        str(score_threshold),
        "--device",
        device,
        "--output-ndjson",
        str(out_ndjson),
        "--summary-json",
        str(out),
    ]
    cfg = SidecarConfig(timeout_s=timeout_s)
    try:
        proc = _sp.run(cmd, capture_output=True, text=True, timeout=cfg.timeout_s)
    except _sp.TimeoutExpired:
        _emit(
            {
                "status": "failed",
                "code": "SIDECAR_TIMEOUT",
                "model_id": model_id,
                "image_dir": str(image_dir),
                "command": cmd,
                "message": f"DEIMv2 benchmark exceeded {cfg.timeout_s}s.",
            },
            out=out,
            fmt=fmt,
        )
        return

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-1000:]
        code = "SIDECAR_COMMAND_FAILED"
        if "no kernel image" in stderr_tail.lower():
            code = "BLACKWELL_SM120_TORCH_INCOMPATIBLE"
        elif "out of memory" in stderr_tail.lower():
            code = "CUDA_OUT_OF_MEMORY"
        _emit(
            {
                "status": "failed",
                "code": code,
                "model_id": model_id,
                "image_dir": str(image_dir),
                "returncode": proc.returncode,
                "stderr_tail": stderr_tail,
            },
            out=out,
            fmt=fmt,
        )
        return

    # Runner already wrote the summary JSON; just emit it.
    payload = json.loads(out.read_text())
    _emit(payload, out=None, fmt=fmt)


__all__ = ["app"]
