# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.21: sidecar-live readiness taxonomy + capability wiring. Weight-free.

These tests never load a model. They assert that the sidecar-live states are
honestly distinct from host-runnable ``*_READY_LIVE`` states, that the precise
decision function promotes a sidecar-live model over its host technical blocker
(but never over a legal/gated/weights block), and that ``model_capabilities()``
surfaces the v3.21 sidecar dimension fields.
"""

from __future__ import annotations

from visionservex.readiness import taxonomy as tx


def test_sidecar_states_registered_and_distinct():
    for state in (
        tx.VLM_READY_LIVE_SIDECAR,
        tx.INFERENCE_READY_LIVE_SIDECAR,
        tx.SEGMENTATION_READY_LIVE_SIDECAR,
    ):
        assert state in tx.READINESS_STATES
        assert state in tx.LIVE_SIDECAR_READY_STATES
        # Sidecar states are deliberately NOT in the host-runnable ready set.
        assert state not in tx.LIVE_READY_STATES


def test_sidecar_states_are_inference_ready_in_coarse_view():
    for state in tx.LIVE_SIDECAR_READY_STATES:
        assert tx.coarse_readiness(state) == "inference-ready"


def test_sidecar_promotion_supersedes_host_dependency_blocker():
    # Florence-2 shape: host dependency-missing, but sidecar-live -> sidecar state.
    state = tx.compute_readiness_state(
        task="vlm",
        implementation_status="implemented",
        engine="florence2",
        engine_registered=True,
        policy_bucket=None,
        license_class="permissive",
        unavailable_reason=None,
        train_ready=False,
        inference_ready=True,
        live_inference_verified=False,
        live_train_verified=False,
        live_inference_blocker="DEPENDENCY_MISSING",
        sidecar_live_verified=True,
    )
    assert state == tx.VLM_READY_LIVE_SIDECAR


def test_sidecar_promotion_never_launders_a_license_block():
    # A copyleft model that a sidecar *could* run must still be LICENSE_BLOCKED.
    state = tx.compute_readiness_state(
        task="vlm",
        implementation_status="implemented",
        engine="x",
        engine_registered=True,
        policy_bucket=None,
        license_class="copyleft",
        unavailable_reason=None,
        train_ready=False,
        inference_ready=True,
        live_inference_verified=False,
        live_train_verified=False,
        sidecar_live_verified=True,
    )
    assert state == tx.LICENSE_BLOCKED


def test_sidecar_promotion_never_overrides_missing_weights():
    state = tx.compute_readiness_state(
        task="pose",
        implementation_status="implemented",
        engine="openmmlab",
        engine_registered=True,
        policy_bucket="not_released_or_unverifiable",
        license_class="permissive",
        unavailable_reason=None,
        train_ready=False,
        inference_ready=True,
        live_inference_verified=False,
        live_train_verified=False,
        sidecar_live_verified=True,
    )
    assert state == tx.WEIGHTS_MISSING


def test_sidecar_task_routing():
    base = {
        "implementation_status": "implemented",
        "engine": "x",
        "engine_registered": True,
        "policy_bucket": None,
        "license_class": "permissive",
        "unavailable_reason": None,
        "train_ready": False,
        "inference_ready": True,
        "live_inference_verified": False,
        "live_train_verified": False,
        "sidecar_live_verified": True,
    }
    assert tx.compute_readiness_state(task="pose", **base) == tx.INFERENCE_READY_LIVE_SIDECAR
    assert tx.compute_readiness_state(task="detect", **base) == tx.INFERENCE_READY_LIVE_SIDECAR
    assert tx.compute_readiness_state(task="segment", **base) == tx.SEGMENTATION_READY_LIVE_SIDECAR
    assert tx.compute_readiness_state(task="vlm", **base) == tx.VLM_READY_LIVE_SIDECAR


def test_anastig_visibility_marks_sidecar():
    v = tx.anastig_visibility(tx.VLM_READY_LIVE_SIDECAR, task="vlm")
    assert v.endswith("_sidecar")
    v2 = tx.anastig_visibility(tx.SEGMENTATION_READY_LIVE_SIDECAR, task="segment")
    assert v2 == "show_segmentation_sidecar"


def test_florence2_capability_is_sidecar_live():
    from visionservex import model_capabilities

    for mid in ("florence-2-base", "florence-2-large"):
        c = model_capabilities(mid)
        assert c["readiness_state"] == tx.VLM_READY_LIVE_SIDECAR
        assert c["readiness"] == "inference-ready"
        assert c["sidecar_supported"] is True
        assert c["sidecar_required"] is True
        assert c["sidecar_name"] == "florence2"
        assert c["sidecar_live"] is True
        assert c["sidecar_cpu_verified"] is True
        assert c["sidecar_gpu_verified"] is False
        assert c["anastig_sidecar_visibility"] == "show_inference_sidecar"
        assert c["blocker"] is None


def test_rtmpose_m_is_openmmlab_sidecar_live():
    from visionservex import model_capabilities

    c = model_capabilities("rtmpose-m")
    assert c["readiness_state"] == tx.INFERENCE_READY_LIVE_SIDECAR
    assert c["readiness"] == "inference-ready"  # coarse view reconciled, not catalog-only
    assert c["sidecar_name"] == "openmmlab"
    assert c["sidecar_required"] and c["sidecar_live"]
    assert c["sidecar_cpu_verified"] and not c["sidecar_gpu_verified"]
    assert c["blocker"] is None


def test_rtmdet_obb_variants_stay_blocked_with_exact_reason():
    # The rotated RTMDet family needs mmrotate (absent from the mmcv-2.x sidecar),
    # so it must NOT be promoted — it stays blocked with a real blocker.
    from visionservex import model_capabilities

    for mid in ("rtmdet-r-s", "rtmdet-r2-s"):
        c = model_capabilities(mid)
        assert c["readiness_state"] not in tx.LIVE_SIDECAR_READY_STATES, mid
        assert c["readiness_state"] not in tx.LIVE_READY_STATES, mid


def test_fine_tune_kind_field_present_and_honest():
    from visionservex import model_capabilities

    # Embedding backbone -> frozen_backbone_head; trainable detector -> full_supervised.
    assert model_capabilities("dinov2-base")["fine_tune_kind"] == "frozen_backbone_head"
    assert model_capabilities("rfdetr-base")["fine_tune_kind"] == "full_supervised"
    # A VLM with no fine-tune path -> none.
    assert model_capabilities("florence-2-base")["fine_tune_kind"] == "none"
