# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.23.0: Universal sidecar manager.

VisionServeX runs on a single host Python with whatever torch the user has.
Some SOTA detectors (DEIMv2, RT-DETRv4) require their OWN Python/torch/CUDA
combination plus custom CUDA ops; they cannot share the host process. The
``SidecarManager`` provides a uniform interface:

- Plan an isolated conda env per sidecar spec.
- Generate exact `conda create` / `pip install` / `git clone` commands
  (default `--dry-run`) so the user can audit before running.
- Optionally execute the create-env flow with a bounded timeout and
  RAM/VRAM/disk pre-check via :mod:`runtime.resource_guard`.
- Run sidecar inference commands via `subprocess.run` with timeout,
  capture STDOUT JSON, separately capture STDERR, kill on timeout, return
  structured :class:`SidecarExecResult`.

No fake success. No raw traceback. No usage error. Every failure carries
a structured ``code`` from the documented blocker set.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SIDECAR_BLOCKER_CODES",
    "SidecarConfig",
    "SidecarExecResult",
    "SidecarManager",
    "SidecarSpec",
]


SIDECAR_BLOCKER_CODES = frozenset(
    [
        "SIDECAR_ENV_MISSING",
        "SIDECAR_CREATE_FAILED",
        "SIDECAR_COMMAND_FAILED",
        "SIDECAR_TIMEOUT",
        "SIDECAR_JSON_MISSING",
        "SIDECAR_JSON_INVALID",
        "CUSTOM_OPS_COMPILATION",
        "CUDA_EXTENSION_BUILD_FAILED",
        "CURRENT_ENV_PY313_TORCH212_UNVALIDATED",
        "TORCH_VERSION_CONFLICT",
        "CHECKPOINT_NOT_FOUND",
        "CHECKPOINT_DOWNLOAD_FAILED",
        "CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP",
        "CONFIG_NOT_FOUND",
        "UPSTREAM_REPO_NOT_FOUND",
        "GATED_AUTH_REQUIRED",
        "LICENSE_RESTRICTION_TRIGGERED",
        "DATASET_LICENSE_UNVERIFIED",
        "RESOURCE_GUARD_BLOCKED",
        "CONDA_NOT_AVAILABLE",
        # v2.24.0: real execution blockers surfaced from sm_120 / Blackwell paths
        "SIDECAR_INSTALL_ROOT_NOT_WRITABLE",
        "BLACKWELL_SM120_TORCH_INCOMPATIBLE",
        "CUDA_KERNEL_LAUNCH_FAILED",
        "GIT_CLONE_FAILED",
        "PIP_INSTALL_FAILED",
        "CONDA_CREATE_FAILED",
        "REQUIREMENTS_TXT_MISSING",
    ]
)


def _resolve_sidecar_root() -> Path:
    """Return the base directory under which sidecar installs are placed.

    Precedence:
    1. ``$VISIONSERVEX_SIDECAR_ROOT`` (explicit override).
    2. ``/opt/visionservex/sidecars`` if ``/opt`` is writable (root/sudo install).
    3. ``~/.cache/visionservex/sidecars`` (user-writable default — picked
       automatically when ``/opt`` is root-owned, which is the normal Linux case).
    """
    override = os.environ.get("VISIONSERVEX_SIDECAR_ROOT")
    if override:
        return Path(override).expanduser()
    opt_parent = Path("/opt")
    if opt_parent.exists() and os.access(opt_parent, os.W_OK):
        return Path("/opt/visionservex/sidecars")
    return Path.home() / ".cache" / "visionservex" / "sidecars"


def _install_root_source() -> str:
    """Return which precedence rule the active install root came from."""
    if os.environ.get("VISIONSERVEX_SIDECAR_ROOT"):
        return "env:VISIONSERVEX_SIDECAR_ROOT"
    opt_parent = Path("/opt")
    if opt_parent.exists() and os.access(opt_parent, os.W_OK):
        return "default:/opt/visionservex/sidecars"
    return "default:~/.cache/visionservex/sidecars"


@dataclass
class SidecarSpec:
    """One sidecar environment specification."""

    name: str
    description: str
    python_version: str
    torch_version: str
    torchvision_version: str
    cuda_channel: str  # e.g. "cu118", "cu124"
    upstream_repo: str
    upstream_branch: str = "main"
    pip_extras: list[str] = field(default_factory=list)
    custom_ops: list[str] = field(default_factory=list)
    license: str = "Apache-2.0"
    notes: str = ""

    @property
    def install_root(self) -> Path:
        return _resolve_sidecar_root() / self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "torchvision_version": self.torchvision_version,
            "cuda_channel": self.cuda_channel,
            "upstream_repo": self.upstream_repo,
            "upstream_branch": self.upstream_branch,
            "pip_extras": list(self.pip_extras),
            "custom_ops": list(self.custom_ops),
            "license": self.license,
            "notes": self.notes,
            "install_root": str(self.install_root),
        }


@dataclass
class SidecarConfig:
    """Per-sidecar runtime config."""

    timeout_s: int = 600
    min_ram_gb: float = 8.0
    min_vram_gb: float = 4.0
    min_disk_gb: float = 20.0


@dataclass
class SidecarExecResult:
    """Result of a single :meth:`SidecarManager.exec` call."""

    status: str  # "ok" | "expected_blocker" | "failed"
    code: str  # blocker code from SIDECAR_BLOCKER_CODES or "OK"
    returncode: int
    sidecar: str
    command: list[str]
    stdout_summary: str = ""
    stderr_summary: str = ""
    json_payload: dict[str, Any] | None = None
    runtime_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "code": self.code,
            "returncode": self.returncode,
            "sidecar": self.sidecar,
            "command": list(self.command),
            "stdout_summary": self.stdout_summary,
            "stderr_summary": self.stderr_summary,
            "json_payload": self.json_payload,
            "runtime_s": round(self.runtime_s, 3),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# Canonical sidecar specs
# ---------------------------------------------------------------------------

_SIDECAR_SPECS: dict[str, SidecarSpec] = {
    "deimv2": SidecarSpec(
        name="deimv2",
        description="DEIMv2 (Real-Time Object Detection Meets DINOv3) — arXiv 2509.20787, Apache-2.0.",
        python_version="3.11.9",
        torch_version="2.5.1",
        torchvision_version="0.20.1",
        cuda_channel="cu124",
        upstream_repo="https://github.com/Intellindust-AI-Lab/DEIMv2.git",
        upstream_branch="main",
        pip_extras=[
            "opencv-python",
            "numpy",
            "pillow",
            "pycocotools",
            "onnxruntime-gpu",
        ],
        custom_ops=["MSDeformableAttention (CUDA extension)"],
        notes=(
            "Custom CUDA ops are compiled on first run; the conda env "
            "must have a matching CUDA toolkit available."
        ),
    ),
    "rtdetrv4": SidecarSpec(
        name="rtdetrv4",
        description=(
            "RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision "
            "Foundation Models — arXiv 2510.25257, Apache-2.0."
        ),
        python_version="3.11.9",
        torch_version="2.5.1",
        torchvision_version="0.20.1",
        cuda_channel="cu124",
        upstream_repo="https://github.com/RT-DETRs/RT-DETRv4.git",
        upstream_branch="main",
        pip_extras=[
            "opencv-python",
            "numpy",
            "pillow",
            "pycocotools",
            "onnxruntime-gpu",
            "faster-coco-eval",
            "transformers",
        ],
        custom_ops=[],
        notes=(
            "Checkpoints are distributed via Google Drive — `pull` returns "
            "CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP with a `gdown` command."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class SidecarManager:
    """Plan and execute sidecar environments."""

    @staticmethod
    def list_specs() -> dict[str, SidecarSpec]:
        return dict(_SIDECAR_SPECS)

    @staticmethod
    def get_spec(name: str) -> SidecarSpec | None:
        return _SIDECAR_SPECS.get(name)

    @staticmethod
    def conda_available() -> bool:
        return shutil.which("conda") is not None or shutil.which("mamba") is not None

    def env_exists(self, name: str) -> bool:
        """Return True if a conda env named ``visionservex-<name>-sidecar`` exists."""
        env_name = f"visionservex-{name}-sidecar"
        if not self.conda_available():
            return False
        try:
            res = subprocess.run(
                ["conda", "env", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            envs = json.loads(res.stdout or "{}").get("envs", [])
            return any(env_name in e for e in envs)
        except Exception:
            return False

    def plan_create(self, spec: SidecarSpec) -> list[str]:
        """Return the ordered list of shell commands that would create the env."""
        env_name = f"visionservex-{spec.name}-sidecar"
        repo_dir = spec.install_root
        index_url = f"https://download.pytorch.org/whl/{spec.cuda_channel}"
        cmds = [
            f"conda create -n {env_name} python={spec.python_version} -y",
            f"git clone -b {spec.upstream_branch} {spec.upstream_repo} {repo_dir}",
            (
                f"conda run -n {env_name} pip install "
                f"torch=={spec.torch_version} torchvision=={spec.torchvision_version} "
                f"--index-url {index_url}"
            ),
            f"conda run -n {env_name} pip install -r {repo_dir}/requirements.txt",
        ]
        if spec.pip_extras:
            cmds.append(f"conda run -n {env_name} pip install " + " ".join(spec.pip_extras))
        return cmds

    def doctor(self, name: str) -> dict[str, Any]:
        """Probe the sidecar environment without modifying anything."""
        spec = self.get_spec(name)
        if spec is None:
            return {
                "status": "failed",
                "code": "SIDECAR_ENV_MISSING",
                "sidecar": name,
                "message": f"Unknown sidecar spec {name!r}. Known: {sorted(_SIDECAR_SPECS)}.",
            }
        conda_ok = self.conda_available()
        env_ok = self.env_exists(name) if conda_ok else False
        repo_ok = spec.install_root.exists() and (spec.install_root / ".git").exists()
        status = "ok" if (conda_ok and env_ok and repo_ok) else "expected_blocker"
        code = (
            "OK"
            if status == "ok"
            else ("CONDA_NOT_AVAILABLE" if not conda_ok else "SIDECAR_ENV_MISSING")
        )
        return {
            "status": status,
            "code": code,
            "sidecar": name,
            "spec": spec.to_dict(),
            "conda_available": conda_ok,
            "env_exists": env_ok,
            "repo_cloned": repo_ok,
            "remediation": (
                "Run `visionservex sidecar create {name} --execute` to install."
                if status != "ok"
                else ""
            ),
        }

    def create(
        self, name: str, *, dry_run: bool = True, config: SidecarConfig | None = None
    ) -> dict[str, Any]:
        """Plan or execute sidecar creation.

        ``dry_run=True`` returns only the planned commands; nothing is changed.
        ``dry_run=False`` runs them with a bounded timeout and resource pre-check.
        """
        spec = self.get_spec(name)
        if spec is None:
            return {
                "status": "failed",
                "code": "SIDECAR_ENV_MISSING",
                "sidecar": name,
                "message": f"Unknown sidecar spec {name!r}.",
            }
        cfg = config or SidecarConfig()
        cmds = self.plan_create(spec)
        if dry_run:
            return {
                "status": "ok",
                "code": "DRY_RUN",
                "sidecar": name,
                "spec": spec.to_dict(),
                "planned_commands": cmds,
                "install_root": str(spec.install_root),
                "install_root_source": _install_root_source(),
                "message": (
                    "Dry-run only. To execute, rerun with --execute (requires conda + ~20 GB disk)."
                ),
            }
        # Real execution path — resource guard pre-check.
        try:
            from visionservex.runtime.resource_guard import enforce_resource_budget

            budget = enforce_resource_budget()
            data = budget.to_dict()
            ram_free = float(data.get("ram_available_gb", 0.0))
            disk_free = float(data.get("disk_free_gb", 0.0))
            if ram_free < cfg.min_ram_gb or disk_free < cfg.min_disk_gb:
                return {
                    "status": "expected_blocker",
                    "code": "RESOURCE_GUARD_BLOCKED",
                    "sidecar": name,
                    "ram_available_gb": ram_free,
                    "disk_free_gb": disk_free,
                    "min_ram_gb": cfg.min_ram_gb,
                    "min_disk_gb": cfg.min_disk_gb,
                    "message": "Sidecar create blocked by resource guard.",
                }
        except Exception:
            # Resource guard module unavailable — proceed cautiously.
            pass

        if not self.conda_available():
            return {
                "status": "expected_blocker",
                "code": "CONDA_NOT_AVAILABLE",
                "sidecar": name,
                "message": "conda not found on PATH; install miniconda first.",
                "planned_commands": cmds,
            }

        # Ensure the install_root parent is writable BEFORE running any command.
        install_root = spec.install_root
        try:
            install_root.parent.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as exc:
            return {
                "status": "failed",
                "code": "SIDECAR_INSTALL_ROOT_NOT_WRITABLE",
                "sidecar": name,
                "install_root": str(install_root),
                "error": str(exc)[:300],
                "message": (
                    f"Cannot create sidecar install root {install_root}. "
                    "Set $VISIONSERVEX_SIDECAR_ROOT to a writable path."
                ),
                "planned_commands": cmds,
            }
        if not os.access(install_root.parent, os.W_OK):
            return {
                "status": "failed",
                "code": "SIDECAR_INSTALL_ROOT_NOT_WRITABLE",
                "sidecar": name,
                "install_root": str(install_root),
                "message": (
                    f"Install-root parent {install_root.parent} is not writable. "
                    "Set $VISIONSERVEX_SIDECAR_ROOT to a writable path."
                ),
                "planned_commands": cmds,
            }

        # Map (command-substring → blocker code) for finer-grained reporting.
        cmd_to_blocker = [
            ("conda create", "CONDA_CREATE_FAILED"),
            ("git clone", "GIT_CLONE_FAILED"),
            ("pip install torch", "PIP_INSTALL_FAILED"),
            ("pip install -r", "PIP_INSTALL_FAILED"),
            ("pip install", "PIP_INSTALL_FAILED"),
        ]

        def _blocker_for(cmd: str) -> str:
            for needle, code in cmd_to_blocker:
                if needle in cmd:
                    return code
            return "SIDECAR_CREATE_FAILED"

        executed: list[dict[str, Any]] = []
        for cmd in cmds:
            t0 = time.time()
            try:
                res = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=cfg.timeout_s,
                )
                executed.append(
                    {
                        "cmd": cmd,
                        "returncode": res.returncode,
                        "elapsed_s": round(time.time() - t0, 2),
                        "stderr_tail": (res.stderr or "")[-400:],
                    }
                )
                if res.returncode != 0:
                    blocker = _blocker_for(cmd)
                    # GIT_CLONE_FAILED when the dir already exists is benign — surface it cleanly.
                    if (
                        blocker == "GIT_CLONE_FAILED"
                        and install_root.exists()
                        and any(install_root.iterdir())
                    ):
                        executed[-1]["note"] = (
                            "Install root already populated; continuing without reclone."
                        )
                        continue
                    return {
                        "status": "failed",
                        "code": blocker,
                        "sidecar": name,
                        "failing_command": cmd,
                        "returncode": res.returncode,
                        "stderr_tail": (res.stderr or "")[-1000:],
                        "executed": executed,
                        "install_root": str(install_root),
                    }
            except subprocess.TimeoutExpired:
                return {
                    "status": "failed",
                    "code": "SIDECAR_TIMEOUT",
                    "sidecar": name,
                    "failing_command": cmd,
                    "timeout_s": cfg.timeout_s,
                    "executed": executed,
                    "install_root": str(install_root),
                }
        return {
            "status": "ok",
            "code": "OK",
            "sidecar": name,
            "executed": executed,
            "install_root": str(install_root),
            "message": "Sidecar environment created.",
        }

    def exec(
        self,
        name: str,
        command: list[str],
        *,
        config: SidecarConfig | None = None,
        input_payload: dict[str, Any] | None = None,
        expect_json: bool = True,
    ) -> SidecarExecResult:
        """Run a command inside the sidecar env, return structured result."""
        spec = self.get_spec(name)
        if spec is None:
            return SidecarExecResult(
                status="failed",
                code="SIDECAR_ENV_MISSING",
                returncode=-1,
                sidecar=name,
                command=command,
                errors=[f"Unknown sidecar {name!r}."],
            )
        cfg = config or SidecarConfig()
        env_name = f"visionservex-{name}-sidecar"

        if not self.conda_available():
            return SidecarExecResult(
                status="expected_blocker",
                code="CONDA_NOT_AVAILABLE",
                returncode=-1,
                sidecar=name,
                command=command,
                errors=["conda not on PATH"],
            )
        if not self.env_exists(name):
            return SidecarExecResult(
                status="expected_blocker",
                code="SIDECAR_ENV_MISSING",
                returncode=-1,
                sidecar=name,
                command=command,
                errors=[f"Conda env {env_name!r} does not exist."],
            )

        # Build the actual subprocess command: `conda run -n <env> <command...>`
        full_cmd = ["conda", "run", "-n", env_name, *command]
        t0 = time.time()
        stdin_data = json.dumps(input_payload) if input_payload is not None else None
        try:
            res = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=cfg.timeout_s,
                input=stdin_data,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return SidecarExecResult(
                status="failed",
                code="SIDECAR_TIMEOUT",
                returncode=-1,
                sidecar=name,
                command=full_cmd,
                runtime_s=time.time() - t0,
                errors=[f"Sidecar timed out after {cfg.timeout_s}s."],
            )

        rc = res.returncode
        stdout = res.stdout or ""
        stderr = res.stderr or ""
        runtime = time.time() - t0

        if rc != 0:
            return SidecarExecResult(
                status="failed",
                code="SIDECAR_COMMAND_FAILED",
                returncode=rc,
                sidecar=name,
                command=full_cmd,
                runtime_s=runtime,
                stdout_summary=stdout[:400],
                stderr_summary=stderr[-400:],
                errors=[stderr[-400:]] if stderr else [],
            )

        if not expect_json:
            return SidecarExecResult(
                status="ok",
                code="OK",
                returncode=rc,
                sidecar=name,
                command=full_cmd,
                runtime_s=runtime,
                stdout_summary=stdout[:400],
                stderr_summary=stderr[-400:],
            )

        # Parse stdout JSON
        try:
            payload = json.loads(stdout.strip()) if stdout.strip() else None
        except json.JSONDecodeError as exc:
            return SidecarExecResult(
                status="failed",
                code="SIDECAR_JSON_INVALID",
                returncode=rc,
                sidecar=name,
                command=full_cmd,
                runtime_s=runtime,
                stdout_summary=stdout[:400],
                stderr_summary=stderr[-400:],
                errors=[f"JSON parse failed: {exc}"],
            )
        if payload is None:
            return SidecarExecResult(
                status="failed",
                code="SIDECAR_JSON_MISSING",
                returncode=rc,
                sidecar=name,
                command=full_cmd,
                runtime_s=runtime,
                stderr_summary=stderr[-400:],
                errors=["Sidecar stdout was empty; expected JSON payload."],
            )
        return SidecarExecResult(
            status="ok",
            code="OK",
            returncode=rc,
            sidecar=name,
            command=full_cmd,
            runtime_s=runtime,
            json_payload=payload,
            stdout_summary=stdout[:200],
            stderr_summary=stderr[-200:],
        )
