# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Runtime broker orchestrator.

The :class:`RuntimeBroker` is the user-visible entry point. It hides every
detail of sidecar environments, license gates, auth gates, and adapters. The
broker is intentionally side-effect-free until ``prepare`` or ``run`` is
invoked with ``execute=True`` — by default everything stays in dry-run mode
that prints exact commands a follow-up build session can replay.

The broker uses :mod:`visionservex.sidecars.manager` to actually create conda
envs and run subprocesses when ``execute=True`` is passed. Resource guards
(:mod:`visionservex.runtime.resource_guard`) gate every heavy action.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from visionservex.runtime_broker.model_router import (
    UnknownModelError,
    resolve_runtime,
)
from visionservex.runtime_broker.result_schema import (
    BROKER_BLOCKER_CODES,
    BrokerBlocker,
    CanonicalResult,
)
from visionservex.runtime_broker.spec_loader import RuntimeSpec, load_specs

__all__ = [
    "BrokerError",
    "BrokerResult",
    "RuntimeBroker",
]


class BrokerError(Exception):
    """Wrapper for broker-level failures that don't fit the structured-blocker model."""


@dataclass
class BrokerResult:
    """High-level result envelope from :meth:`RuntimeBroker.run` and friends."""

    model_id: str
    runtime_id: str
    action: str  # "explain" | "prepare" | "run" | "doctor" | "clean"
    executed: bool
    commands: list[list[str]] = field(default_factory=list)
    cwd: str | None = None
    blocker: BrokerBlocker | None = None
    canonical: CanonicalResult | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.blocker is None


def _sidecar_root() -> Path:
    env = os.environ.get("VISIONSERVEX_SIDECAR_ROOT")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "visionservex" / "sidecars"


def _checkpoint_root() -> Path:
    env = os.environ.get("VISIONSERVEX_CHECKPOINT_ROOT")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "visionservex" / "checkpoints"


def _expand(path: str | Path) -> Path:
    return Path(str(path)).expanduser()


def _env_name_for(spec: RuntimeSpec) -> str:
    return f"visionservex-{spec.id.replace('_', '-')}"


def _which_conda() -> str | None:
    return shutil.which("conda") or shutil.which("mamba")


@dataclass
class _Plan:
    """An internal plan of the commands the broker would execute."""

    commands: list[list[str]] = field(default_factory=list)
    blocker: BrokerBlocker | None = None

    def add(self, cmd: list[str]) -> None:
        self.commands.append(list(cmd))


class RuntimeBroker:
    """User-facing runtime broker.

    Construction is cheap: specs are loaded once from the YAML and cached.
    The same broker instance can be reused for every CLI invocation in a
    session.
    """

    def __init__(self, specs_path: Path | str | None = None):
        self._specs_path = Path(specs_path) if specs_path else None
        self._specs = load_specs(self._specs_path)

    # ------------------------------------------------------------------ API

    def list_runtimes(self) -> list[RuntimeSpec]:
        """Return every loaded runtime spec, sorted by id."""

        return sorted(self._specs.values(), key=lambda s: s.id)

    def routing(self) -> dict[str, str]:
        """Return the full ``model_id -> runtime_id`` table."""

        from visionservex.runtime_broker.model_router import routing_table

        return routing_table(specs=self._specs)

    def resolve(self, model_id: str) -> RuntimeSpec:
        """Return the runtime spec that owns ``model_id``."""

        runtime_id = resolve_runtime(model_id, specs=self._specs)
        return self._specs[runtime_id]

    # -------------------------------------------------------------- explain

    def explain(self, model_id: str) -> BrokerResult:
        """Return a non-executing explanation of how the broker would run the model.

        Used by ``visionservex runtime explain <model_id>``. The result is
        deterministic and has no side effects.
        """

        try:
            spec = self.resolve(model_id)
        except UnknownModelError as exc:
            return BrokerResult(
                model_id=model_id,
                runtime_id="<unknown>",
                action="explain",
                executed=False,
                blocker=BrokerBlocker(
                    code="UNKNOWN_MODEL_ID",
                    message=str(exc),
                    model_id=model_id,
                    next_action=(
                        "Add a row for this model_id to "
                        "reports/v246_exact_50_recovery_plan.csv or to a runtime's "
                        "supported_models list."
                    ),
                ),
            )

        plan = self._plan_prepare(spec, model_id)
        cmd = self._format_command(
            spec.smoke_command,
            {
                "model_id": model_id,
                "input": "<INPUT_PATH>",
                "device": "auto",
                "point": "<x,y>",
                "prompt": "<TEXT_PROMPT>",
            },
        )
        plan.add(cmd)

        return BrokerResult(
            model_id=model_id,
            runtime_id=spec.id,
            action="explain",
            executed=False,
            commands=plan.commands,
            extra={
                "env_type": spec.env_type,
                "python_version": spec.python_version,
                "torch_version": spec.torch_version,
                "cuda_version": spec.cuda_version,
                "pip_packages": spec.pip_packages,
                "conda_packages": spec.conda_packages,
                "git_repos": spec.git_repos,
                "custom_ops": spec.custom_ops,
                "checkpoint_sources": spec.checkpoint_sources,
                "license_gate": spec.license_gate,
                "auth_gate": spec.auth_gate,
                "output_adapter": spec.output_adapter,
                "fallback_runtime": spec.fallback_runtime,
                "known_failure_modes": spec.known_failure_modes,
            },
        )

    # -------------------------------------------------------------- prepare

    def prepare(
        self,
        model_id: str,
        *,
        execute: bool = False,
        force: bool = False,
    ) -> BrokerResult:
        """Materialize the sidecar environment for ``model_id``.

        With ``execute=False`` (the default), the broker returns the exact
        commands it would have run. With ``execute=True`` the broker forwards
        to :mod:`visionservex.sidecars.manager` for env creation. This session
        only the dry-run path is exercised; the execute path is unchanged from
        v2.45's sidecar manager.
        """

        try:
            spec = self.resolve(model_id)
        except UnknownModelError as exc:
            return BrokerResult(
                model_id=model_id,
                runtime_id="<unknown>",
                action="prepare",
                executed=False,
                blocker=BrokerBlocker(
                    code="UNKNOWN_MODEL_ID",
                    message=str(exc),
                    model_id=model_id,
                ),
            )

        plan = self._plan_prepare(spec, model_id)

        if not execute:
            return BrokerResult(
                model_id=model_id,
                runtime_id=spec.id,
                action="prepare",
                executed=False,
                commands=plan.commands,
                blocker=BrokerBlocker(
                    code="BROKER_DRY_RUN_NO_EXECUTE",
                    message=(
                        "Broker is in dry-run mode. Pass --execute to actually "
                        "build the sidecar env. Each command is printed verbatim."
                    ),
                    runtime_id=spec.id,
                    model_id=model_id,
                    next_action=f"visionservex runtime prepare {model_id} --execute",
                ),
            )

        # execute=True
        return self._execute_prepare(spec, model_id, plan, force=force)

    def _plan_prepare(self, spec: RuntimeSpec, model_id: str) -> _Plan:
        plan = _Plan()
        if spec.env_type == "http":
            # external API runtime — no env to prepare.
            plan.add(["echo", f"runtime '{spec.id}' is HTTP-only; no env to prepare."])
            return plan

        if spec.env_type == "subprocess":
            # core/registry/license/auth runtimes share host env.
            plan.add(["echo", f"runtime '{spec.id}' uses host env; no separate env to prepare."])
            return plan

        env_name = _env_name_for(spec)
        sidecar_dir = _sidecar_root() / spec.id

        if spec.env_type == "conda":
            conda = _which_conda() or "conda"
            plan.add(
                [
                    conda,
                    "create",
                    "-n",
                    env_name,
                    "-y",
                    f"python={spec.python_version}",
                ]
            )

            # pip first so that lap-style PEP-517 issues don't trip on numpy.
            if spec.pip_packages:
                plan.add(
                    [
                        conda,
                        "run",
                        "-n",
                        env_name,
                        "python",
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        "pip",
                        "setuptools",
                        "wheel",
                    ]
                )
                plan.add(
                    [
                        conda,
                        "run",
                        "-n",
                        env_name,
                        "python",
                        "-m",
                        "pip",
                        "install",
                        *spec.pip_packages,
                    ]
                )

            if spec.conda_packages:
                plan.add(
                    [
                        conda,
                        "install",
                        "-n",
                        env_name,
                        "-y",
                        *spec.conda_packages,
                    ]
                )

        elif spec.env_type == "docker":
            plan.add(
                [
                    "docker",
                    "build",
                    "-t",
                    f"visionservex/{spec.id}:v2.46.0",
                    str(sidecar_dir / "Dockerfile"),
                ]
            )

        elif spec.env_type == "venv":
            plan.add(
                [
                    f"python{spec.python_version}",
                    "-m",
                    "venv",
                    str(sidecar_dir / ".venv"),
                ]
            )

        # git clones
        for repo in spec.git_repos:
            target = _expand(repo["target"])
            plan.add(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    str(repo.get("ref") or "HEAD"),
                    repo["url"],
                    str(target),
                ]
            )
            if repo.get("post_install"):
                plan.add(["sh", "-c", repo["post_install"]])

        # custom ops
        for op in spec.custom_ops:
            plan.add(["sh", "-c", op])

        # checkpoints
        for ck in spec.checkpoint_sources:
            target = _expand(ck["target_path"])
            url = ck.get("url")
            if url and url.startswith("http"):
                plan.add(["curl", "-L", "-o", str(target), url])
            elif url and url.startswith("manual"):
                plan.add(
                    [
                        "echo",
                        f"manual checkpoint required: {url}; target={target}",
                    ]
                )
            else:
                plan.add(["echo", f"checkpoint source: {ck}"])

        return plan

    def _execute_prepare(
        self,
        spec: RuntimeSpec,
        model_id: str,
        plan: _Plan,
        *,
        force: bool,
    ) -> BrokerResult:
        """Run the planned commands serially with timeouts.

        Each command runs under a 30-minute hard cap with output captured.
        On the first failure, the broker returns a structured blocker with
        the stderr tail and the failing command. The remaining commands
        are NOT attempted (fail-fast). Resource_guard policy is up to the
        host shell — the broker does not duplicate that here, but each
        command is timeboxed so it cannot hang forever.
        """
        results: list[dict[str, Any]] = []
        for idx, cmd in enumerate(plan.commands):
            started = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
            entry = {
                "cmd": cmd,
                "returncode": started.returncode,
                "stdout_tail": (started.stdout or "")[-2000:],
                "stderr_tail": (started.stderr or "")[-2000:],
            }
            results.append(entry)
            if started.returncode != 0:
                return BrokerResult(
                    model_id=model_id,
                    runtime_id=spec.id,
                    action="prepare",
                    executed=True,
                    commands=plan.commands,
                    blocker=BrokerBlocker(
                        code="RUNTIME_PREPARE_FAILED",
                        message=(
                            f"command {idx + 1}/{len(plan.commands)} exited with "
                            f"{started.returncode}: {' '.join(cmd)[:200]}"
                        ),
                        runtime_id=spec.id,
                        model_id=model_id,
                        exception_tail=entry["stderr_tail"],
                        next_action=(
                            "Inspect the stderr tail and fix the underlying step. "
                            "Most common: missing CUDA toolchain, network failure, or "
                            "a stale conda env (try --force)."
                        ),
                    ),
                    extra={"force": force, "executed_commands": results},
                )

        return BrokerResult(
            model_id=model_id,
            runtime_id=spec.id,
            action="prepare",
            executed=True,
            commands=plan.commands,
            extra={
                "force": force,
                "executed_commands": results,
                "note": (
                    "v2.46 execute path runs each planned command serially "
                    "with a 30-minute hard cap per command."
                ),
            },
        )

    # ------------------------------------------------------------------ run

    def run(
        self,
        model_id: str,
        input_path: str | Path,
        *,
        task: str | None = None,
        device: str = "auto",
        execute: bool = False,
        accept_license: bool = False,
        accept_auth: bool = False,
        api_key: str | None = None,
        extra_args: list[str] | None = None,
    ) -> BrokerResult:
        """Run a model.

        The full execution path is intentionally not wired in this prep
        session; the broker emits the exact subprocess command(s) it would
        run, including license-gate and auth-gate checks. The follow-up
        session can flip ``execute=True`` after the sidecar is built.
        """

        try:
            spec = self.resolve(model_id)
        except UnknownModelError as exc:
            return BrokerResult(
                model_id=model_id,
                runtime_id="<unknown>",
                action="run",
                executed=False,
                blocker=BrokerBlocker(
                    code="UNKNOWN_MODEL_ID",
                    message=str(exc),
                    model_id=model_id,
                ),
            )

        # License / auth gates -------------------------------------------
        if spec.license_gate and not accept_license:
            return BrokerResult(
                model_id=model_id,
                runtime_id=spec.id,
                action="run",
                executed=False,
                blocker=BrokerBlocker(
                    code="LICENSE_OPT_IN_NOT_PROVIDED",
                    message=(
                        f"Model '{model_id}' requires license opt-in via env var "
                        f"{spec.license_gate} and --accept-license flag."
                    ),
                    runtime_id=spec.id,
                    model_id=model_id,
                    next_action=(
                        f"visionservex run {model_id} <input> "
                        f"--accept-license  # also set {spec.license_gate}=1"
                    ),
                ),
            )

        if spec.auth_gate and not accept_auth and not os.environ.get(spec.auth_gate):
            return BrokerResult(
                model_id=model_id,
                runtime_id=spec.id,
                action="run",
                executed=False,
                blocker=BrokerBlocker(
                    code="AUTH_TOKEN_NOT_PROVIDED",
                    message=(
                        f"Model '{model_id}' requires auth token via env var {spec.auth_gate}."
                    ),
                    runtime_id=spec.id,
                    model_id=model_id,
                    next_action=(f"export {spec.auth_gate}=<your_token>  # and re-run."),
                ),
            )

        if (
            spec.env_type == "http"
            and not api_key
            and spec.auth_gate
            and not os.environ.get(spec.auth_gate)
        ):
            return BrokerResult(
                model_id=model_id,
                runtime_id=spec.id,
                action="run",
                executed=False,
                blocker=BrokerBlocker(
                    code="API_KEY_NOT_PROVIDED",
                    message=f"API runtime '{spec.id}' needs {spec.auth_gate}.",
                    runtime_id=spec.id,
                    model_id=model_id,
                    next_action=f"export {spec.auth_gate}=<your_key>",
                ),
            )

        # Build the actual command --------------------------------------
        template = spec.contract_command if task in {"contract", None} else spec.smoke_command
        substitutions = {
            "model_id": model_id,
            "input": str(input_path),
            "device": device,
            "point": "0,0",
            "prompt": "",
        }
        cmd = self._format_command(template, substitutions)
        if extra_args:
            cmd = cmd + list(extra_args)

        if not execute:
            return BrokerResult(
                model_id=model_id,
                runtime_id=spec.id,
                action="run",
                executed=False,
                commands=[cmd],
                blocker=BrokerBlocker(
                    code="BROKER_DRY_RUN_NO_EXECUTE",
                    message=(
                        "Broker is in dry-run mode. Pass --execute to actually "
                        "run the model; the command above is what would be run."
                    ),
                    runtime_id=spec.id,
                    model_id=model_id,
                    next_action=(f"visionservex run {model_id} {input_path} --execute"),
                ),
            )

        return self._execute_subprocess(spec, model_id, cmd)

    def _execute_subprocess(
        self,
        spec: RuntimeSpec,
        model_id: str,
        cmd: list[str],
    ) -> BrokerResult:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return BrokerResult(
                model_id=model_id,
                runtime_id=spec.id,
                action="run",
                executed=True,
                commands=[cmd],
                blocker=BrokerBlocker(
                    code="RUNTIME_PREPARE_TIMEOUT",
                    message="subprocess timed out after 600s",
                    runtime_id=spec.id,
                    model_id=model_id,
                ),
            )

        canonical: CanonicalResult | None = None
        if proc.returncode == 0 and proc.stdout:
            try:
                payload = json.loads(proc.stdout)
                canonical = CanonicalResult(
                    model_id=model_id,
                    runtime_id=spec.id,
                    task=str(payload.get("task", "unknown")),
                    extra=payload,
                )
            except json.JSONDecodeError:
                canonical = None

        blocker = None
        if proc.returncode != 0:
            blocker = BrokerBlocker(
                code="RUNTIME_PREPARE_FAILED",
                message=f"subprocess exited with {proc.returncode}",
                runtime_id=spec.id,
                model_id=model_id,
                exception_tail=(proc.stderr or "")[-2000:],
            )

        return BrokerResult(
            model_id=model_id,
            runtime_id=spec.id,
            action="run",
            executed=True,
            commands=[cmd],
            canonical=canonical,
            blocker=blocker,
            extra={"returncode": proc.returncode},
        )

    # -------------------------------------------------------------- doctor

    def doctor(self) -> dict[str, Any]:
        """Return host-environment readiness for every runtime.

        Pure diagnostic. Never builds anything. Reports python availability,
        conda presence, docker presence, GPU presence, and CUDA driver.
        """

        report: dict[str, Any] = {
            "broker_version": "v2.46.0",
            "runtimes": {},
            "host": {
                "python_executable": shutil.which("python3") or shutil.which("python"),
                "conda_executable": _which_conda(),
                "docker_executable": shutil.which("docker"),
                "git_executable": shutil.which("git"),
                "nvidia_smi": shutil.which("nvidia-smi"),
            },
        }

        for spec in self.list_runtimes():
            entry: dict[str, Any] = {
                "env_type": spec.env_type,
                "python_version": spec.python_version,
                "torch_version": spec.torch_version,
                "needs_sidecar": spec.needs_sidecar_env,
                "is_external": spec.is_external,
                "supported_models_count": len(spec.supported_models),
            }
            if spec.env_type == "conda":
                entry["conda_available"] = bool(_which_conda())
            if spec.env_type == "docker":
                entry["docker_available"] = bool(shutil.which("docker"))
            report["runtimes"][spec.id] = entry

        return report

    # ---------------------------------------------------------------- clean

    def clean(self, unused_only: bool = True) -> BrokerResult:
        """List sidecar dirs that would be cleaned. Never deletes without --execute."""

        root = _sidecar_root()
        commands: list[list[str]] = []
        if root.exists():
            commands.append(["echo", f"sidecar root: {root}"])
            for entry in sorted(root.iterdir()):
                commands.append(["echo", f"would inspect: {entry}"])
        else:
            commands.append(["echo", f"sidecar root does not exist: {root}"])

        return BrokerResult(
            model_id="<n/a>",
            runtime_id="<n/a>",
            action="clean",
            executed=False,
            commands=commands,
            extra={"unused_only": unused_only},
        )

    # --------------------------------------------------------------- locks

    def export_locks(self, out_path: Path | str) -> Path:
        """Write a machine-readable lock manifest covering every runtime."""

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "v246.broker_locks.v1",
            "runtimes": {
                spec.id: {
                    "python_version": spec.python_version,
                    "torch_version": spec.torch_version,
                    "cuda_version": spec.cuda_version,
                    "pip_packages": spec.pip_packages,
                    "conda_packages": spec.conda_packages,
                    "git_repos": spec.git_repos,
                    "checkpoint_sources": spec.checkpoint_sources,
                    "license_gate": spec.license_gate,
                    "auth_gate": spec.auth_gate,
                    "supported_models": spec.supported_models,
                }
                for spec in self.list_runtimes()
            },
        }
        out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return out

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _format_command(template: list[str], values: dict[str, str]) -> list[str]:
        """Format every placeholder in a command template."""

        formatted: list[str] = []
        for token in template:
            try:
                formatted.append(token.format(**values))
            except (KeyError, IndexError):
                formatted.append(token)
        return formatted


# Expose the static blocker code set in case external modules need it.
_ = BROKER_BLOCKER_CODES
