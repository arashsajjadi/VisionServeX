# SPDX-License-Identifier: Apache-2.0
"""Broker dry-run path must produce structured commands without side effects."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from visionservex.runtime_broker import (
    BrokerBlocker,
    RuntimeBroker,
)


def test_explain_unknown_model_returns_structured_blocker() -> None:
    result = RuntimeBroker().explain("not-a-real-model-name-zzz")
    assert isinstance(result.blocker, BrokerBlocker)
    assert result.blocker.code == "UNKNOWN_MODEL_ID"


def test_prepare_without_execute_emits_dry_run_blocker() -> None:
    result = RuntimeBroker().prepare("internimage-t", execute=False)
    assert result.executed is False
    assert result.blocker is not None
    assert result.blocker.code == "BROKER_DRY_RUN_NO_EXECUTE"
    # Must contain the conda create command:
    flat = "\n".join(" ".join(c) for c in result.commands)
    assert "conda" in flat
    assert "python=3.10" in flat


def test_run_dryrun_includes_input_path_in_command() -> None:
    result = RuntimeBroker().run(
        "co-dino-inst-vit-l-coco",
        Path("/tmp/example.jpg"),
        task="contract",
        execute=False,
    )
    assert result.blocker is not None
    assert result.blocker.code == "BROKER_DRY_RUN_NO_EXECUTE"
    flat = "\n".join(" ".join(c) for c in result.commands)
    assert "/tmp/example.jpg" in flat
    assert "co-dino-inst-vit-l-coco" in flat


def test_license_gate_blocks_without_opt_in() -> None:
    result = RuntimeBroker().run(
        "yolo11x.pt",
        Path("/tmp/example.jpg"),
        execute=True,
        accept_license=False,
    )
    assert result.blocker is not None
    assert result.blocker.code == "LICENSE_OPT_IN_NOT_PROVIDED"


def test_license_gate_passes_with_opt_in() -> None:
    # With opt-in but no execute the broker emits the dry-run blocker.
    # With opt-in AND execute the broker will try to run the (placeholder)
    # license-gate subprocess; we don't care that the subprocess succeeds,
    # just that we got past the license gate. Use execute=False to stay safe.
    result = RuntimeBroker().run(
        "yolo11x.pt",
        Path("/tmp/example.jpg"),
        execute=False,
        accept_license=True,
    )
    assert result.blocker is not None
    assert result.blocker.code == "BROKER_DRY_RUN_NO_EXECUTE"


def test_auth_gate_blocks_without_hf_token() -> None:
    # sam3-base requires HF_TOKEN. Clear it for the duration of the test.
    saved = os.environ.pop("HF_TOKEN", None)
    try:
        result = RuntimeBroker().run(
            "sam3-base",
            Path("/tmp/example.jpg"),
            execute=False,
        )
        assert result.blocker is not None
        assert result.blocker.code == "AUTH_TOKEN_NOT_PROVIDED"
    finally:
        if saved is not None:
            os.environ["HF_TOKEN"] = saved


def test_list_runtimes_has_15_plus_external_api() -> None:
    items = RuntimeBroker().list_runtimes()
    ids = [s.id for s in items]
    # 15 required + external_api_runtime = 16 minimum
    assert len(ids) >= 16


def test_export_locks_writes_machine_readable_manifest(tmp_path: Path) -> None:
    out = tmp_path / "locks.json"
    path = RuntimeBroker().export_locks(out)
    assert path.exists()
    import json

    payload = json.loads(path.read_text())
    assert "runtimes" in payload
    assert "core_py311" in payload["runtimes"]
    assert "license_gate_runtime" in payload["runtimes"]


def test_doctor_returns_host_section() -> None:
    report = RuntimeBroker().doctor()
    assert "host" in report
    assert "runtimes" in report
    assert "conda_executable" in report["host"]


@pytest.mark.parametrize(
    "model_id",
    [
        "deim-m",
        "deim-s",
        "oneformer-convnext-large",
        "agriclip",
        "dinov3-vitb16",
        "prithvi-eo-2.0",
    ],
)
def test_registry_and_permissive_models_use_core_runtime(model_id: str) -> None:
    spec = RuntimeBroker().resolve(model_id)
    assert spec.id == "core_py311"
