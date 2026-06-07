# SPDX-License-Identifier: Apache-2.0
"""v2.47 test: called_in_notebook / covered_by_notebook / execution_origin semantics."""

from __future__ import annotations

from visionservex.reporting.v239_reconciler import (
    _HEALTHY_STATES_V247,
    _RESTRICTED_LICENSE_MODELS,
    _derive_command_attempted,
    _derive_covered_by_notebook,
    _derive_execution_origin,
)

# ---------------------------------------------------------------------------
# _derive_covered_by_notebook
# ---------------------------------------------------------------------------


def test_healthy_states_are_covered() -> None:
    for state in _HEALTHY_STATES_V247:
        assert _derive_covered_by_notebook(state, "", False, False) is True


def test_license_gated_states_are_covered() -> None:
    for state in (
        "opt_in_license_required",
        "license_blocked",
        "auth_required",
        "external_api_only",
    ):
        assert _derive_covered_by_notebook(state, "", False, False) is True


def test_sidecar_required_without_evidence_is_not_covered() -> None:
    assert _derive_covered_by_notebook("sidecar_required", "", False, False) is False


def test_current_rerun_is_covered() -> None:
    assert _derive_covered_by_notebook("sidecar_required", "current_rerun", False, False) is True


def test_historical_validated_is_covered() -> None:
    assert (
        _derive_covered_by_notebook("sidecar_required", "historical_validated", False, False)
        is True
    )


def test_historical_fallback_is_covered() -> None:
    assert _derive_covered_by_notebook("sidecar_required", "", True, False) is True


def test_called_in_notebook_overrides_sidecar() -> None:
    assert _derive_covered_by_notebook("sidecar_required", "", False, True) is True


# ---------------------------------------------------------------------------
# _derive_command_attempted
# ---------------------------------------------------------------------------


def test_sidecar_command_includes_model_and_runtime() -> None:
    cmd = _derive_command_attempted("bytetrack", "sidecar_required", "", "tracking_bytetrack_py310")
    assert "bytetrack" in cmd
    assert "--execute" in cmd
    assert "tracking_bytetrack_py310" in cmd


def test_license_command_yolo_uses_accept_agpl() -> None:
    cmd = _derive_command_attempted(
        "yolo11x.pt", "opt_in_license_required", "OPT_IN_LICENSE_REQUIRED", ""
    )
    assert "--accept-agpl" in cmd


def test_license_command_pml_uses_accept_pml() -> None:
    cmd = _derive_command_attempted(
        "rfdetr-seg-xlarge", "opt_in_license_required", "OPT_IN_LICENSE_REQUIRED", ""
    )
    assert "--accept-pml" in cmd


def test_api_command_includes_api_key() -> None:
    cmd = _derive_command_attempted("dino-x-api", "external_api_only", "EXTERNAL_API_REQUIRED", "")
    assert "api-key" in cmd.lower() or "API_KEY" in cmd


def test_auth_command_uses_auth_flag() -> None:
    cmd = _derive_command_attempted("sam3-base", "auth_required", "HF_AUTH_REQUIRED", "")
    assert "auth" in cmd.lower()


def test_wired_command_is_contract() -> None:
    cmd = _derive_command_attempted("agriclip", "wired", "", "core_py311")
    assert "agriclip" in cmd


def test_checkpoint_required_command_includes_pull() -> None:
    cmd = _derive_command_attempted("deimv2-n", "checkpoint_required", "CHECKPOINT_REQUIRED", "")
    assert "deimv2-n" in cmd


def test_command_not_blank_for_all_typical_states() -> None:
    test_cases = [
        ("bytetrack", "sidecar_required", "", "tracking_bytetrack_py310"),
        ("deimv2-n", "checkpoint_required", "", ""),
        ("yolo11x.pt", "opt_in_license_required", "", ""),
        ("dino-x-api", "external_api_only", "", ""),
        ("sam3-base", "auth_required", "", ""),
        ("deim-m", "upstream_deprecated", "", ""),
        ("oneformer-convnext-large", "wired", "", ""),
        ("agriclip", "wired", "", ""),
    ]
    for mid, state, blocker, runtime in test_cases:
        cmd = _derive_command_attempted(mid, state, blocker, runtime)
        assert cmd.strip(), f"command_attempted is blank for {mid} / {state}"


# ---------------------------------------------------------------------------
# _derive_execution_origin
# ---------------------------------------------------------------------------


def test_historical_origin_from_metric_origin() -> None:
    assert (
        _derive_execution_origin("smoke_passed", "", "historical_validated", False, "")
        == "historical_validated"
    )


def test_current_run_executed_when_called() -> None:
    assert (
        _derive_execution_origin("smoke_passed", "", "current_rerun", True, "")
        == "current_run_executed"
    )


def test_current_run_status_gate_when_not_called() -> None:
    assert (
        _derive_execution_origin("smoke_passed", "", "current_rerun", False, "")
        == "current_run_status_gate"
    )


def test_api_origin() -> None:
    assert (
        _derive_execution_origin("external_api_only", "", "", False, "") == "external_api_required"
    )


def test_auth_origin() -> None:
    assert _derive_execution_origin("auth_required", "", "", False, "") == "auth_required"


def test_alias_origin() -> None:
    assert (
        _derive_execution_origin("wired", "", "", False, "alias_resolved_to_deimv2-m")
        == "registry_alias"
    )


# ---------------------------------------------------------------------------
# _RESTRICTED_LICENSE_MODELS
# ---------------------------------------------------------------------------


def test_restricted_license_models_set_has_expected_count() -> None:
    assert len(_RESTRICTED_LICENSE_MODELS) == 14


def test_ultralytics_yolo_in_restricted() -> None:
    for m in ("yolo11x.pt", "yolo26x.pt", "yolov10b.pt", "yolov8x.pt"):
        assert m in _RESTRICTED_LICENSE_MODELS


def test_fastsam_in_restricted() -> None:
    assert "fastsam-s" in _RESTRICTED_LICENSE_MODELS
    assert "fastsam-x" in _RESTRICTED_LICENSE_MODELS


def test_rfdetr_seg_xl_in_restricted() -> None:
    assert "rfdetr-seg-xlarge" in _RESTRICTED_LICENSE_MODELS
    assert "rfdetr-seg-2xlarge" in _RESTRICTED_LICENSE_MODELS


def test_totalsegmentator_in_restricted() -> None:
    assert "totalsegmentator" in _RESTRICTED_LICENSE_MODELS
