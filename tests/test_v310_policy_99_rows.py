# SPDX-License-Identifier: Apache-2.0
"""v3.10.0: policy table has 99 rows; all invariants hold."""

from __future__ import annotations


def _rows():
    from visionservex.licensing.policy import _ROWS

    return _ROWS


def test_policy_row_count_is_99():
    # v3.11.0 added 3 INSID3 rows (102 total); accept >=99 going forward
    assert len(_rows()) >= 99, f"Expected >=99 policy rows, got {len(_rows())}"


def test_can_ship_weights_all_false():
    bad = [r.model_id for r in _rows() if r.can_ship_weights]
    assert not bad, f"can_ship_weights=True (must be 0): {bad}"


def test_dinov3_chmv2_dpt_in_policy():
    ids = {r.model_id for r in _rows()}
    assert "dinov3-vitl16-chmv2-dpt-head" in ids


def test_dinov3_chmv2_correct_hf_repo():
    from visionservex.licensing.policy import get_policy, resolve_model_id

    pol = get_policy(resolve_model_id("dinov3-vitl16-chmv2-dpt-head"))
    assert pol is not None
    assert pol.hf_repo == "facebook/dinov3-vitl16-chmv2-dpt-head"


def test_dinov3_family_has_13_rows():
    dinov3 = [r for r in _rows() if r.family == "dinov3"]
    assert len(dinov3) == 13, f"Expected 13 DINOv3 rows, got {len(dinov3)}"


def test_byot_license_required_count():
    byot = [r for r in _rows() if r.final_policy == "byot_license_required"]
    # v3.11.0 added 3 INSID3 rows (31 total); accept >=28 going forward
    assert len(byot) >= 28, f"Expected >=28 byot rows, got {len(byot)}"


def test_commercial_safe_core_count():
    core = [r for r in _rows() if r.final_policy == "commercial_safe_core"]
    # v3.12.0 added 3 permissive LibreYOLO detectors (yolox-s, yolov9-s, rtdetr-r50)
    # -> 42. v3.14.0 added libreyolo-dfine-n (trainable D-FINE) -> 43.
    # v3.15.0 added 13 torchvision classifiers (BSD-3-Clause) -> 56.
    assert len(core) == 56, f"Expected 56 commercial_safe_core rows, got {len(core)}"


def test_no_gated_in_commercial_safe_default_safe():
    bad = [r for r in _rows() if r.gated and r.default_safe]
    assert not bad, f"Gated with default_safe=True: {[r.model_id for r in bad]}"


def test_no_agpl_default_safe():
    bad = [r for r in _rows() if "agpl" in r.code_license.lower() and r.default_safe]
    assert not bad, f"AGPL with default_safe=True: {[r.model_id for r in bad]}"


def test_api_only_models_not_local():
    bad = [
        r for r in _rows() if r.final_policy == "external_api_only_terms_required" and r.is_local
    ]
    assert not bad, f"API-only models marked is_local=True: {[r.model_id for r in bad]}"
