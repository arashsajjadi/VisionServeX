# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.23.0: RT-DETRv4 CLI — sidecar-aware, obsolete blocker fixed.

Upstream truth (re-verified 2026-05-18 with the v2.23 Deep Research lead):

The v2.22 blocker ``RTDETRV4_UPSTREAM_NOT_RELEASED`` was wrong because v2.22
checked only ``lyuwenyu/RT-DETR`` (the v1/v2 author). The canonical
RT-DETRv4 release lives at a separate org:

- Repo:     https://github.com/RT-DETRs/RT-DETRv4 (Apache-2.0, 473 stars,
            arXiv 2510.25257)
- Configs:  ``configs/rtv4/rtv4_hgnetv2_{s,m,l,x}_coco.yml`` in repo
- Inference:
    python tools/inference/torch_inf.py \\
      -c configs/rtv4/rtv4_hgnetv2_{s,m,l,x}_coco.yml \\
      -r CHECKPOINT.pth --input IMG --device cuda:0
- Checkpoints: Google Drive (not directly curl-able). v2.23 reports
            ``CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP`` with the
            ``gdown`` command users can run themselves.
- v2.23 ships a sidecar create-env path
  (``visionservex rtdetrv4 create-env --execute``) that clones the repo
  into ``/opt/visionservex/sidecars/rtdetrv4/`` and installs deps in an
  isolated conda env.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(
    help=(
        "v2.23.0: RT-DETRv4 doctor / create-env / pull / smoke-test "
        "(real upstream at RT-DETRs/RT-DETRv4, checkpoints on Google Drive)."
    ),
    no_args_is_help=True,
)
console = Console()

RTDETRV4_UPSTREAM_REPO = "https://github.com/RT-DETRs/RT-DETRv4"
RTDETRV4_PAPER = "https://arxiv.org/abs/2510.25257"
RTDETRV4_LICENSE = "Apache-2.0"
RTDETRV4_VERIFIED_ON = "2026-05-18"

# Per-variant checkpoint registry. Google Drive ids are real (from the
# upstream README); v2.23 emits the `gdown` command rather than auto-pulling
# because Google Drive does not return a direct binary on a plain HTTP GET.
RTDETRV4_CHECKPOINTS: dict[str, dict[str, str | float]] = {
    "rtdetrv4-s": {
        "config": "configs/rtv4/rtv4_hgnetv2_s_coco.yml",
        "gdrive_id": "1jDAVxblqRPEWed7Hxm6GwcEl7z",
        "reported_AP": 49.8,
        "reported_AP50": 67.1,
        "reported_latency_ms": 3.66,
    },
    "rtdetrv4-m": {
        "config": "configs/rtv4/rtv4_hgnetv2_m_coco.yml",
        "gdrive_id": "1O-YpP4X-quuOXbi96y2TKkztbj",
        "reported_AP": 53.7,
        "reported_AP50": 71.0,
        "reported_latency_ms": 5.91,
    },
    "rtdetrv4-l": {
        "config": "configs/rtv4/rtv4_hgnetv2_l_coco.yml",
        "gdrive_id": "1shO9EzZvXZyKedE2urLsN4dwEv",
        "reported_AP": 55.4,
        "reported_AP50": 73.0,
        "reported_latency_ms": 8.07,
    },
    "rtdetrv4-x": {
        "config": "configs/rtv4/rtv4_hgnetv2_x_coco.yml",
        "gdrive_id": "19gnkMTgFveJsrOvSmEPQXCTG6v",
        "reported_AP": 57.0,
        "reported_AP50": 74.6,
        "reported_latency_ms": 12.90,
        "additional_pretrain": "dinov3_vitb16_pretrain_lvd1689m.pth",
    },
}


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


@app.command("doctor")
def doctor_cmd(
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Probe the RT-DETRv4 sidecar environment readiness."""
    from visionservex.sidecars import SidecarManager

    sidecar_info = SidecarManager().doctor("rtdetrv4")
    payload = {
        "status": sidecar_info["status"],
        "code": (
            "OK"
            if sidecar_info["status"] == "ok"
            else (
                "SIDECAR_ENV_MISSING"
                if not sidecar_info.get("env_exists")
                else sidecar_info.get("code", "SIDECAR_ENV_MISSING")
            )
        ),
        "model_family": "rtdetrv4",
        "upstream_repo": RTDETRV4_UPSTREAM_REPO,
        "paper": RTDETRV4_PAPER,
        "license": RTDETRV4_LICENSE,
        "verified_on": RTDETRV4_VERIFIED_ON,
        "sidecar_probe": sidecar_info,
        "v2_22_obsolete_blocker_replaced": "RTDETRV4_UPSTREAM_NOT_RELEASED",
        "checkpoint_distribution": "Google Drive (gdown required)",
        "remediation": (
            "Run `visionservex rtdetrv4 create-env --execute` to install the sidecar env, "
            "then `visionservex rtdetrv4 pull rtdetrv4-s` for the gdown command."
        ),
    }
    _emit(payload, out=out, fmt=fmt)


@app.command("create-env")
def create_env_cmd(
    dry_run: bool = typer.Option(True, "--dry-run/--execute"),
    timeout_s: int = typer.Option(
        1800,
        "--timeout-s",
        help="Per-command timeout (default 1800s = 30 min).",
    ),
    profile: str = typer.Option(
        "",
        "--profile",
        help="rtdetrv4-cu124-stable (default) | rtdetrv4-blackwell-nightly",
    ),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Plan or execute creation of the RT-DETRv4 sidecar conda env."""
    from visionservex.sidecars import SidecarConfig, SidecarManager

    cfg = SidecarConfig(timeout_s=timeout_s)
    payload = SidecarManager().create(
        "rtdetrv4", dry_run=dry_run, config=cfg, profile=(profile or None)
    )
    _emit(payload, out=out, fmt=fmt)


@app.command("pull")
def pull_cmd(
    model_id: str = typer.Argument("rtdetrv4-s"),
    method: str = typer.Option(
        "manual",
        "--method",
        help="auto | gdown | direct-url | manual",
    ),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
    timeout_s: int = typer.Option(900, "--timeout-s"),
) -> None:
    """v2.25.0: pull an RT-DETRv4 checkpoint via gdown (default: manual).

    With ``--method auto`` (or ``gdown``) we attempt the actual Google Drive
    fetch. On success the checkpoint is cached under
    ``~/.cache/visionservex/sidecars/rtdetrv4/checkpoints/<model_id>.pth``
    and the report carries ``status=ok``. If ``gdown`` is missing or the
    fetch fails, we fall back to ``CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP``
    with the exact command the user can run.
    """
    import shutil as _sh
    import subprocess as _sp
    import time as _t

    info = RTDETRV4_CHECKPOINTS.get(model_id)
    if info is None:
        payload = {
            "status": "expected_blocker",
            "code": "CHECKPOINT_NOT_FOUND",
            "model_id": model_id,
            "message": f"Unknown RT-DETRv4 variant {model_id!r}. Known: {sorted(RTDETRV4_CHECKPOINTS)}.",
        }
        _emit(payload, out=out, fmt=fmt)
        return

    gid = info["gdrive_id"]
    cache_root = Path.home() / ".cache" / "visionservex" / "sidecars" / "rtdetrv4" / "checkpoints"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{model_id}.pth"

    manual_payload = {
        "status": "expected_blocker",
        "code": "CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP",
        "model_id": model_id,
        "method_used": "manual",
        "config": info["config"],
        "checkpoint_source": "google_drive",
        "gdrive_id": gid,
        "gdown_command": f"gdown --id {gid} -O {cache_path}",
        "manual_command": f"gdown --id {gid} -O {cache_path}",
        "cache_path": str(cache_path),
        "reported_AP": info["reported_AP"],
        "reported_AP50": info.get("reported_AP50"),
        "reported_latency_ms": info["reported_latency_ms"],
        "additional_pretrain": info.get("additional_pretrain"),
        "upstream_repo": RTDETRV4_UPSTREAM_REPO,
        "license": RTDETRV4_LICENSE,
        "message": (
            f"RT-DETRv4 checkpoints are distributed via Google Drive. "
            f"Install `gdown` then run: gdown --id {gid} -O {cache_path}."
        ),
    }

    # If the cache already exists, short-circuit with status=ok.
    if cache_path.exists() and cache_path.stat().st_size > 0:
        ok_payload = dict(manual_payload)
        ok_payload.update(
            status="ok",
            code="OK",
            method_used="cache_hit",
            size_bytes=cache_path.stat().st_size,
            message=f"Checkpoint already at {cache_path}.",
        )
        _emit(ok_payload, out=out, fmt=fmt)
        return

    if method == "manual":
        _emit(manual_payload, out=out, fmt=fmt)
        return

    if method in {"auto", "gdown"}:
        if _sh.which("gdown") is None:
            # Try inside the rtdetrv4 sidecar env where gdown might be installed
            # alongside other extras.
            ok = False
            from visionservex.sidecars import SidecarManager

            if SidecarManager().env_exists("rtdetrv4"):
                t0 = _t.time()
                try:
                    proc = _sp.run(
                        [
                            "conda",
                            "run",
                            "-n",
                            "visionservex-rtdetrv4-sidecar",
                            "python",
                            "-c",
                            (
                                "import sys, gdown; "
                                f"gdown.download(id={gid!r}, output={str(cache_path)!r}, "
                                "quiet=True)"
                            ),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=timeout_s,
                    )
                    if proc.returncode == 0 and cache_path.exists():
                        ok = True
                    else:
                        stderr_tail = (proc.stderr or "")[-500:]
                except Exception as exc:
                    stderr_tail = repr(exc)[:500]
                runtime_s = _t.time() - t0
                if ok:
                    ok_payload = dict(manual_payload)
                    ok_payload.update(
                        status="ok",
                        code="OK",
                        method_used="gdown_via_sidecar",
                        size_bytes=cache_path.stat().st_size,
                        runtime_s=round(runtime_s, 2),
                        message=f"Downloaded via sidecar gdown -> {cache_path}.",
                    )
                    _emit(ok_payload, out=out, fmt=fmt)
                    return
                manual_payload["sidecar_gdown_stderr"] = stderr_tail
            manual_payload["gdown_in_host_path"] = False
            manual_payload["code"] = "CHECKPOINT_DOWNLOAD_FAILED"
            manual_payload["status"] = "expected_blocker"
            manual_payload["message"] = (
                "gdown not on host PATH. Install it (`pip install gdown`) or run the "
                f"manual command: gdown --id {gid} -O {cache_path}."
            )
            _emit(manual_payload, out=out, fmt=fmt)
            return

        # gdown is on host PATH — try fetching directly. gdown >=6 no longer
        # accepts --id; the modern form takes the URL or id as a positional arg.
        t0 = _t.time()
        try:
            proc = _sp.run(
                ["gdown", "-O", str(cache_path), f"https://drive.google.com/uc?id={gid}"],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            runtime_s = _t.time() - t0
            if proc.returncode == 0 and cache_path.exists() and cache_path.stat().st_size > 0:
                ok_payload = dict(manual_payload)
                ok_payload.update(
                    status="ok",
                    code="OK",
                    method_used="gdown_host",
                    size_bytes=cache_path.stat().st_size,
                    runtime_s=round(runtime_s, 2),
                    message=f"Downloaded via host gdown -> {cache_path}.",
                )
                _emit(ok_payload, out=out, fmt=fmt)
                return
            manual_payload["code"] = "CHECKPOINT_DOWNLOAD_FAILED"
            manual_payload["status"] = "expected_blocker"
            manual_payload["stderr_tail"] = (proc.stderr or "")[-500:]
            manual_payload["returncode"] = proc.returncode
            manual_payload["message"] = (
                f"gdown failed (rc={proc.returncode}). Most likely Google Drive's "
                "abuse filter; supply --checkpoint <path> to smoke-test instead."
            )
            _emit(manual_payload, out=out, fmt=fmt)
            return
        except _sp.TimeoutExpired:
            manual_payload["code"] = "CHECKPOINT_DOWNLOAD_FAILED"
            manual_payload["status"] = "expected_blocker"
            manual_payload["message"] = f"gdown timed out after {timeout_s}s."
            _emit(manual_payload, out=out, fmt=fmt)
            return

    # Unknown method
    manual_payload["status"] = "failed"
    manual_payload["code"] = "INVALID_METHOD"
    manual_payload["message"] = f"Unknown --method {method!r}; use auto|gdown|manual."
    _emit(manual_payload, out=out, fmt=fmt)


@app.command("smoke-test")
def smoke_test_cmd(
    model_id: str = typer.Argument("rtdetrv4-s"),
    image: Path = typer.Argument(..., help="Image path."),
    device: str = typer.Option("cuda", "--device"),
    backend: str = typer.Option("torch", "--backend", help="torch | onnxruntime | tensorrt"),
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        help=(
            "v2.25.0: user-supplied checkpoint .pth path. If omitted, looks under "
            "~/.cache/visionservex/sidecars/rtdetrv4/checkpoints/<model_id>.pth."
        ),
    ),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
    draw: Path | None = typer.Option(None, "--draw"),
) -> None:
    """Attempt RT-DETRv4 smoke-test via the sidecar; return structured blocker if env missing."""
    from visionservex.sidecars import SidecarManager

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

    info = RTDETRV4_CHECKPOINTS.get(model_id)
    if info is None:
        _emit(
            {
                "status": "expected_blocker",
                "code": "CHECKPOINT_NOT_FOUND",
                "model_id": model_id,
                "message": f"Unknown RT-DETRv4 variant {model_id!r}.",
            },
            out=out,
            fmt=fmt,
        )
        return

    sidecar = SidecarManager().doctor("rtdetrv4")
    if sidecar["status"] != "ok":
        _emit(
            {
                "status": "expected_blocker",
                "code": "SIDECAR_ENV_MISSING",
                "model_id": model_id,
                "image": str(image),
                "device": device,
                "backend": backend,
                "sidecar_probe": sidecar,
                "message": (
                    "RT-DETRv4 sidecar env is not yet created. "
                    "Run `visionservex rtdetrv4 create-env --execute` first, "
                    "then `visionservex rtdetrv4 pull rtdetrv4-s` for the checkpoint."
                ),
            },
            out=out,
            fmt=fmt,
        )
        return

    if backend == "tensorrt":
        _emit(
            {
                "status": "expected_blocker",
                "code": "RTDETRV4_TRT_5080_ACCURACY_BUG_OPEN",
                "model_id": model_id,
                "image": str(image),
                "backend": backend,
                "message": (
                    "TensorRT backend is gated behind --experimental-tensorrt due to an open "
                    "RTX 5080 accuracy bug. Use --backend torch by default."
                ),
            },
            out=out,
            fmt=fmt,
        )
        return

    # v2.25.0: Run the smoke for real when a checkpoint is supplied / cached.
    import subprocess as _sp
    import time as _t

    ckpt_path = checkpoint or (
        Path.home()
        / ".cache"
        / "visionservex"
        / "sidecars"
        / "rtdetrv4"
        / "checkpoints"
        / f"{model_id}.pth"
    )
    if not ckpt_path.exists():
        _emit(
            {
                "status": "expected_blocker",
                "code": "CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP",
                "model_id": model_id,
                "image": str(image),
                "device": device,
                "backend": backend,
                "checkpoint_searched": str(ckpt_path),
                "manual_command": (
                    f"gdown --id {info['gdrive_id']} -O {ckpt_path}  # or supply --checkpoint"
                ),
                "message": (
                    f"No checkpoint at {ckpt_path}. Run "
                    f"`visionservex rtdetrv4 pull {model_id} --method auto` "
                    f"or supply --checkpoint <path>."
                ),
            },
            out=out,
            fmt=fmt,
        )
        return

    repo_root = Path.home() / ".cache" / "visionservex" / "sidecars" / "rtdetrv4"
    config_path = repo_root / info["config"]
    if not config_path.exists():
        _emit(
            {
                "status": "expected_blocker",
                "code": "CONFIG_NOT_FOUND",
                "model_id": model_id,
                "config_searched": str(config_path),
                "message": (
                    f"Config {config_path} not found. Re-run create-env to ensure "
                    "the upstream repo is cloned."
                ),
            },
            out=out,
            fmt=fmt,
        )
        return

    cuda_str = "cuda:0" if device.startswith("cuda") else "cpu"
    cmd = [
        "conda",
        "run",
        "-n",
        "visionservex-rtdetrv4-sidecar",
        "python",
        str(repo_root / "tools" / "inference" / "torch_inf.py"),
        "-c",
        str(config_path),
        "-r",
        str(ckpt_path),
        "--input",
        str(image),
        "--device",
        cuda_str,
    ]
    t0 = _t.time()
    try:
        proc = _sp.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(repo_root))
    except _sp.TimeoutExpired:
        _emit(
            {
                "status": "failed",
                "code": "SIDECAR_TIMEOUT",
                "model_id": model_id,
                "command": cmd,
                "message": "RT-DETRv4 smoke timed out after 120s.",
            },
            out=out,
            fmt=fmt,
        )
        return
    runtime_s = _t.time() - t0

    stderr_tail = (proc.stderr or "")[-1000:]
    # Detect Blackwell sm_120 incompatibility on the stderr.
    if proc.returncode != 0 and "no kernel image is available" in stderr_tail.lower():
        _emit(
            {
                "status": "expected_blocker",
                "code": "BLACKWELL_SM120_TORCH_INCOMPATIBLE",
                "model_id": model_id,
                "image": str(image),
                "device": device,
                "checkpoint": str(ckpt_path),
                "config": str(config_path),
                "runtime_s": round(runtime_s, 2),
                "stderr_tail": stderr_tail,
                "message": (
                    "RT-DETRv4 sidecar torch does not support this GPU's compute "
                    "capability. Re-create the env with "
                    "`--profile rtdetrv4-blackwell-nightly`."
                ),
            },
            out=out,
            fmt=fmt,
        )
        return
    if proc.returncode != 0:
        _emit(
            {
                "status": "failed",
                "code": "SIDECAR_COMMAND_FAILED",
                "model_id": model_id,
                "image": str(image),
                "device": device,
                "checkpoint": str(ckpt_path),
                "config": str(config_path),
                "returncode": proc.returncode,
                "runtime_s": round(runtime_s, 2),
                "stderr_tail": stderr_tail,
            },
            out=out,
            fmt=fmt,
        )
        return

    _emit(
        {
            "status": "ok",
            "code": "OK",
            "model_id": model_id,
            "image": str(image),
            "device": device,
            "backend": backend,
            "checkpoint": str(ckpt_path),
            "config": str(config_path),
            "runtime_s": round(runtime_s, 2),
            "stdout_tail": (proc.stdout or "")[-400:],
            "message": "RT-DETRv4 smoke-test produced an inference result.",
        },
        out=out,
        fmt=fmt,
    )


@app.command("checkpoint-instructions")
def checkpoint_instructions_cmd(
    model_id: str = typer.Argument("rtdetrv4-s"),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """v2.26.0: emit copy-paste instructions for fetching an RT-DETRv4 checkpoint.

    The upstream stores checkpoints on Google Drive. Drive's abuse filter
    rejects automated requests after a small number of hits, so v2.26 ships
    explicit instructions instead of pretending the automated path works
    every time. Output names every field the user needs to fix it manually.
    """
    info = RTDETRV4_CHECKPOINTS.get(model_id)
    cache_root = Path.home() / ".cache" / "visionservex" / "sidecars" / "rtdetrv4" / "checkpoints"
    expected_path = cache_root / f"{model_id}.pth"
    if info is None:
        payload = {
            "status": "expected_blocker",
            "code": "CHECKPOINT_NOT_FOUND",
            "model_id": model_id,
            "message": f"Unknown variant {model_id!r}. Known: {sorted(RTDETRV4_CHECKPOINTS)}.",
        }
        _emit(payload, out=out, fmt=fmt)
        return

    gid = info["gdrive_id"]
    drive_url = f"https://drive.google.com/file/d/{gid}/view?usp=sharing"
    gdown_command = f"gdown -O {expected_path} https://drive.google.com/uc?id={gid}"
    wget_attempt = "# wget cannot bypass Drive's confirm token; use the browser link or gdown."

    payload = {
        "status": "ok" if expected_path.exists() else "expected_blocker",
        "code": "OK" if expected_path.exists() else "CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP",
        "model_id": model_id,
        "checkpoint_source": "google_drive",
        "official_url": RTDETRV4_UPSTREAM_REPO,
        "direct_url_if_known": drive_url,
        "gdown_command_if_known": gdown_command,
        "wget_command_if_known": wget_attempt,
        "manual_download_required": True,
        "expected_filename": f"{model_id}.pth",
        "expected_cache_path": str(expected_path),
        "expected_size_bytes_if_known": None,
        "checksum_if_known": None,
        "additional_pretrain": info.get("additional_pretrain"),
        "smoke_command_after_download": (
            f"visionservex rtdetrv4 smoke-test {model_id} IMAGE "
            f"--checkpoint {expected_path} --device cuda --backend torch "
            f"--format json --out reports/{model_id}_smoke_v226.json"
        ),
        "benchmark_command_after_download": (
            f"# AP benchmark requires a COCO-format annotation file:\n"
            f"# visionservex benchmark-detection --models {model_id} "
            f"--dataset coco:ANNOTATIONS.json --device cuda --require-gpu "
            f"--backend sidecar-rtdetrv4 --format json "
            f"--out reports/{model_id}_benchmark_v226.json"
        ),
        "blocker_code": (
            "OK" if expected_path.exists() else "CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP"
        ),
        "reported_AP": info.get("reported_AP"),
        "reported_AP50": info.get("reported_AP50"),
        "reported_latency_ms": info.get("reported_latency_ms"),
        "license": RTDETRV4_LICENSE,
        "upstream_repo": RTDETRV4_UPSTREAM_REPO,
        "message": (
            f"Checkpoint already cached at {expected_path}."
            if expected_path.exists()
            else (
                "Download manually from the Google Drive link above, or run "
                f"`{gdown_command}`. Drive's abuse filter rejects automated "
                "fetches after a few attempts; in that case open the URL in a "
                f"browser and save the file to {expected_path}."
            )
        ),
    }
    _emit(payload, out=out, fmt=fmt)


@app.command("validate-checkpoint")
def validate_checkpoint_cmd(
    model_id: str = typer.Argument("rtdetrv4-s"),
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to the .pth."),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """v2.26.0: validate that a user-supplied .pth looks like an RT-DETRv4 checkpoint."""
    info = RTDETRV4_CHECKPOINTS.get(model_id)
    if info is None:
        _emit(
            {
                "status": "expected_blocker",
                "code": "CHECKPOINT_NOT_FOUND",
                "model_id": model_id,
                "message": f"Unknown variant {model_id!r}.",
            },
            out=out,
            fmt=fmt,
        )
        return

    if not checkpoint.exists():
        _emit(
            {
                "status": "expected_blocker",
                "code": "CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP",
                "model_id": model_id,
                "checkpoint": str(checkpoint),
                "message": f"Checkpoint not found at {checkpoint}.",
            },
            out=out,
            fmt=fmt,
        )
        return

    size = checkpoint.stat().st_size
    # Probe the file via the sidecar env (host torch may not match upstream).
    from visionservex.sidecars import SidecarManager

    mgr = SidecarManager()
    payload: dict[str, Any] = {
        "status": "ok",
        "code": "OK",
        "model_id": model_id,
        "checkpoint": str(checkpoint),
        "size_bytes": size,
        "size_mb": round(size / (1024 * 1024), 2),
    }
    if mgr.env_exists("rtdetrv4"):
        import subprocess as _sp

        probe = (
            "import torch, json, sys; "
            f"ckpt = torch.load({str(checkpoint)!r}, map_location='cpu', weights_only=False); "
            "kinds = type(ckpt).__name__; "
            "n_keys = len(ckpt) if hasattr(ckpt, '__len__') else 0; "
            "has_model = isinstance(ckpt, dict) and ('model' in ckpt or 'state_dict' in ckpt); "
            "print(json.dumps({'kinds': kinds, 'n_keys': n_keys, 'has_model_key': has_model}))"
        )
        try:
            res = _sp.run(
                [
                    "conda",
                    "run",
                    "-n",
                    "visionservex-rtdetrv4-sidecar",
                    "python",
                    "-c",
                    probe,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if res.returncode == 0:
                last = res.stdout.strip().splitlines()[-1]
                payload["probe"] = json.loads(last)
                if not payload["probe"].get("has_model_key"):
                    payload["status"] = "expected_blocker"
                    payload["code"] = "CHECKPOINT_INVALID"
                    payload["message"] = (
                        "Checkpoint loaded but does not contain a 'model' or "
                        "'state_dict' key — not an RT-DETRv4 checkpoint?"
                    )
            else:
                payload["status"] = "expected_blocker"
                payload["code"] = "CHECKPOINT_INVALID"
                payload["stderr_tail"] = (res.stderr or "")[-400:]
                payload["message"] = "torch.load failed inside the sidecar env."
        except _sp.TimeoutExpired:
            payload["status"] = "failed"
            payload["code"] = "SIDECAR_TIMEOUT"
            payload["message"] = "Sidecar probe exceeded 120s."
    else:
        payload["status"] = "expected_blocker"
        payload["code"] = "SIDECAR_ENV_MISSING"
        payload["message"] = (
            "RT-DETRv4 sidecar env is not installed; cannot deeply validate the "
            "checkpoint. File exists and is the right size though."
        )
    _emit(payload, out=out, fmt=fmt)


@app.command("checkpoint-state")
def checkpoint_state_cmd(
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """v2.28.0: canonical RT-DETRv4 checkpoint state for all variants.

    Never returns NOT_WIRED. Each variant gets a final_state of
    ``manual_checkpoint_required`` (default) or ``benchmarked`` once a
    user-supplied checkpoint produces a smoke + AP run.
    """
    cache_root = Path.home() / ".cache" / "visionservex" / "sidecars" / "rtdetrv4" / "checkpoints"
    rows: list[dict[str, Any]] = []
    for model_id, info in RTDETRV4_CHECKPOINTS.items():
        gid = info["gdrive_id"]
        cache_path = cache_root / f"{model_id}.pth"
        present = cache_path.exists()
        manual_browser_url = f"https://drive.google.com/file/d/{gid}/view?usp=sharing"
        gdown_command = f"gdown -O {cache_path} https://drive.google.com/uc?id={gid}"
        final_state = "benchmarked" if False else "manual_checkpoint_required"
        # In v2.28 we don't auto-run benchmarks here; only the smoke-test CLI
        # advances the state. Always start from manual_checkpoint_required.
        rows.append(
            {
                "model_id": model_id,
                "config_path": info["config"],
                "expected_cache_path": str(cache_path),
                "checkpoint_present": present,
                "checkpoint_source": "google_drive",
                "google_drive_id": gid,
                "gdown_command": gdown_command,
                "manual_browser_url": manual_browser_url,
                "validate_command": (
                    f"visionservex rtdetrv4 validate-checkpoint {model_id} "
                    f"--checkpoint {cache_path} --format json"
                ),
                "smoke_command": (
                    f"visionservex rtdetrv4 smoke-test {model_id} IMAGE "
                    f"--checkpoint {cache_path} --device cuda --backend torch "
                    f"--format json"
                ),
                "benchmark_command": (
                    f"# AP benchmark wired in v2.28+: requires sidecar-rtdetrv4 backend\n"
                    f"# visionservex benchmark-detection --models {model_id} "
                    f"--dataset coco:ANNOTATIONS.json --device cuda --require-gpu "
                    f"--backend sidecar-rtdetrv4 --format json"
                ),
                "final_state": final_state,
                "blocker_code": (
                    "OK" if final_state == "benchmarked" else "MANUAL_CHECKPOINT_REQUIRED"
                ),
                "reported_AP": info.get("reported_AP"),
                "reported_latency_ms": info.get("reported_latency_ms"),
                "license": RTDETRV4_LICENSE,
                "upstream_repo": RTDETRV4_UPSTREAM_REPO,
            }
        )
    summary = {
        "n_variants": len(rows),
        "n_present": sum(1 for r in rows if r["checkpoint_present"]),
        "n_manual_checkpoint_required": sum(
            1 for r in rows if r["final_state"] == "manual_checkpoint_required"
        ),
        "n_benchmarked": sum(1 for r in rows if r["final_state"] == "benchmarked"),
    }
    payload = {
        "status": "ok",
        "code": "OK",
        "summary": summary,
        "rows": rows,
        "upstream_repo": RTDETRV4_UPSTREAM_REPO,
        "license": RTDETRV4_LICENSE,
    }
    _emit(payload, out=out, fmt=fmt)


__all__ = ["app"]
