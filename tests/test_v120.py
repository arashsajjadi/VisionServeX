# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tests for v1.2.0: accuracy taxonomy, new D-FINE IDs, experimental SOTA, benchmark commands."""

from __future__ import annotations

import json

import pytest
from PIL import Image
from typer.testing import CliRunner

from visionservex.cli.main import app
from visionservex.registry import default_registry

runner = CliRunner()


# ============================================================
# Phase 1 — Model taxonomy (model_category field)
# ============================================================


def test_model_category_field_exists_on_all_entries():
    """Every registry entry must have a model_category."""
    for e in default_registry().list():
        assert e.model_category is not None, f"{e.id} missing model_category"


def test_demo_fast_models():
    reg = default_registry()
    demo_fast = [e for e in reg.list() if e.model_category == "demo_fast"]
    ids = {e.id for e in demo_fast}
    assert "dfine-n" in ids, "dfine-n should be demo_fast"
    assert "rfdetr-nano" in ids, "rfdetr-nano should be demo_fast"
    assert "grounding-dino-tiny" in ids
    assert "rfdetr-seg-nano" in ids


def test_accuracy_grade_detection_models():
    reg = default_registry()
    acc = [e for e in reg.list(task="detect") if e.model_category == "accuracy_grade"]
    ids = {e.id for e in acc}
    # Objects365+COCO D-FINE variants must be accuracy_grade
    assert "dfine-s-o365-coco" in ids
    assert "dfine-m-o365-coco" in ids
    assert "dfine-l-o365-coco" in ids
    assert "dfine-x-o365-coco" in ids
    # Legacy aliases too
    assert "dfine-s" in ids
    assert "dfine-m" in ids
    # RF-DETR accuracy variants
    assert "rfdetr-base" in ids
    assert "rfdetr-medium" in ids
    assert "rfdetr-large" in ids


def test_production_recommended_models():
    reg = default_registry()
    pr = [e for e in reg.list() if e.model_category == "production_recommended"]
    ids = {e.id for e in pr}
    assert "rfdetr-small" in ids
    assert "rfdetr-seg-small" in ids
    assert "swinv2-tiny" in ids
    assert "sam-vit-base" in ids


def test_experimental_sota_models():
    reg = default_registry()
    exp = [e for e in reg.list() if e.model_category == "experimental_sota"]
    ids = {e.id for e in exp}
    assert "deim-s" in ids
    assert "deimv2-s" in ids
    assert "rtdetrv4-s" in ids
    assert "rtdetrv4-m" in ids


def test_utility_models():
    reg = default_registry()
    util = [e for e in reg.list() if e.model_category == "utility"]
    ids = {e.id for e in util}
    assert "mock-detect" in ids
    assert "mock-classify" in ids


def test_expert_sidecar_models():
    reg = default_registry()
    es = [e for e in reg.list() if e.model_category == "expert_sidecar"]
    ids = {e.id for e in es}
    assert "internimage-t" in ids
    assert "rtmpose-s" in ids


def test_unavailable_with_reason_models():
    reg = default_registry()
    ua = [e for e in reg.list() if e.model_category == "unavailable_with_reason"]
    ids = {e.id for e in ua}
    assert "rfdetr-seg-large" in ids
    # All unavailable entries must have either unavailable_reason or notes
    for e in ua:
        assert e.unavailable_reason or e.notes, f"{e.id} missing explanation"


def test_model_category_type_valid():
    """model_category values must be valid ModelCategory literals.

    v1.6.0 added more categories — read the live Literal so tests stay in sync.
    """
    import typing

    from visionservex.registry.registry import ModelCategory

    valid = set(typing.get_args(ModelCategory))
    for e in default_registry().list():
        assert e.model_category in valid, f"{e.id} has invalid model_category={e.model_category!r}"


# ============================================================
# Phase 2 — D-FINE new model IDs
# ============================================================


def test_dfine_coco_ids_in_registry():
    reg = default_registry()
    for mid in ["dfine-n-coco", "dfine-s-coco", "dfine-m-coco", "dfine-l-coco", "dfine-x-coco"]:
        e = reg.get(mid)
        assert e.task == "detect"
        assert e.engine == "dfine"
        assert e.family == "dfine"
        assert e.hf_repo_id is not None


def test_dfine_o365coco_ids_in_registry():
    reg = default_registry()
    for mid in ["dfine-s-o365-coco", "dfine-m-o365-coco", "dfine-l-o365-coco", "dfine-x-o365-coco"]:
        e = reg.get(mid)
        assert e.task == "detect"
        assert e.engine == "dfine"
        assert e.model_category == "accuracy_grade"
        assert e.implementation_status == "wired"
        assert e.license == "Apache-2.0"


def test_dfine_o365coco_correct_hf_repos():
    reg = default_registry()
    assert reg.get("dfine-s-o365-coco").hf_repo_id == "ustc-community/dfine-small-obj2coco"
    assert reg.get("dfine-m-o365-coco").hf_repo_id == "ustc-community/dfine-medium-obj2coco"
    assert reg.get("dfine-l-o365-coco").hf_repo_id == "ustc-community/dfine-large-obj2coco-e25"
    assert reg.get("dfine-x-o365-coco").hf_repo_id == "ustc-community/dfine-xlarge-obj2coco"


def test_dfine_n_coco_same_repo_as_dfine_n():
    reg = default_registry()
    assert reg.get("dfine-n-coco").hf_repo_id == reg.get("dfine-n").hf_repo_id


def test_dfine_engine_maps_new_ids():
    """dfine engine _HF_REPOS must include all new model IDs."""
    from visionservex.engines.dfine import _HF_REPOS

    for mid in [
        "dfine-n-coco",
        "dfine-s-coco",
        "dfine-s-o365-coco",
        "dfine-m-o365-coco",
        "dfine-l-o365-coco",
        "dfine-x-o365-coco",
    ]:
        assert mid in _HF_REPOS, f"_HF_REPOS missing {mid!r}"


def test_dfine_s_o365coco_auto_download():
    e = default_registry().get("dfine-s-o365-coco")
    assert e.auto_download is True, "dfine-s-o365-coco should be auto_download=true"


def test_dfine_x_o365coco_requires_gpu():
    e = default_registry().get("dfine-x-o365-coco")
    assert "cuda" in e.supported_devices
    assert e.auto_download is False, "dfine-x-o365-coco is large; auto_download should be false"


# ============================================================
# Phase 3 — RF-DETR model categorisation
# ============================================================


def test_rfdetr_small_production_recommended():
    e = default_registry().get("rfdetr-small")
    assert e.model_category == "production_recommended"
    assert e.implementation_status == "wired"


def test_rfdetr_medium_accuracy_grade():
    e = default_registry().get("rfdetr-medium")
    assert e.model_category == "accuracy_grade"


def test_rfdetr_seg_nano_demo_fast():
    e = default_registry().get("rfdetr-seg-nano")
    assert e.model_category == "demo_fast"
    assert "not_good_for" in e.model_dump() or e.not_good_for


def test_rfdetr_seg_large_unavailable():
    e = default_registry().get("rfdetr-seg-large")
    assert e.model_category == "unavailable_with_reason"
    assert e.implementation_status == "stub"


# ============================================================
# Phase 4 — Experimental SOTA candidates
# ============================================================


def test_deim_entries_experimental():
    reg = default_registry()
    for mid in ["deim-s", "deim-m", "deimv2-s", "deimv2-m"]:
        e = reg.get(mid)
        assert e.model_category == "experimental_sota"
        assert e.implementation_status == "stub"
        assert e.status == "experimental"
        assert e.unavailable_reason, f"{mid} must have unavailable_reason"
        assert e.task == "detect"
        assert "cuda" in e.supported_devices


def test_rtdetrv4_entries_experimental():
    reg = default_registry()
    for mid in ["rtdetrv4-s", "rtdetrv4-m", "rtdetrv4-l", "rtdetrv4-x"]:
        e = reg.get(mid)
        assert e.model_category == "experimental_sota"
        assert e.implementation_status == "stub"
        assert e.task == "detect"
        assert e.license_uncertain is True


def test_experimental_not_auto_download():
    reg = default_registry()
    for mid in ["deim-s", "rtdetrv4-s"]:
        e = reg.get(mid)
        assert e.auto_download is False, f"{mid} should not auto_download"


# ============================================================
# Phase 5 — Segmentation: MaskDINO
# ============================================================


def test_maskdino_entries_in_registry():
    reg = default_registry()
    for mid in ["maskdino-r50-coco", "maskdino-r50-panoptic"]:
        e = reg.get(mid)
        assert e.model_category == "experimental_sota"
        assert e.implementation_status == "stub"
        assert e.task == "segment"
        assert e.unavailable_reason


# ============================================================
# Phase 8 — benchmark-competitiveness CLI
# ============================================================


def test_benchmark_competitiveness_mock_json():
    """benchmark-competitiveness with mock-detect should run without real weights."""
    result = runner.invoke(
        app,
        [
            "benchmark",
            "benchmark-competitiveness",
            "--models",
            "mock-detect",
            "--max-images",
            "3",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "models" in payload
    assert "conclusion" in payload
    assert payload["benchmark_type"] == "competitiveness_latency_and_detection_health"


def test_benchmark_competitiveness_output_schema():
    result = runner.invoke(
        app,
        [
            "benchmark",
            "benchmark-competitiveness",
            "--models",
            "mock-detect",
            "--max-images",
            "5",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["images_tested"] == 5
    model_result = payload["models"][0]
    assert "latency_p50_ms" in model_result
    assert "avg_detections" in model_result
    assert "zero_detection_rate" in model_result
    assert "invalid_box_rate" in model_result


def test_benchmark_competitiveness_invalid_model_skipped():
    """Non-detect task models should be skipped gracefully."""
    result = runner.invoke(
        app,
        [
            "benchmark",
            "benchmark-competitiveness",
            "--models",
            "mock-detect,mock-classify",
            "--max-images",
            "3",
            "--json",
        ],
    )
    # Should succeed (mock-detect runs, mock-classify skipped or included but non-detect)
    assert result.exit_code == 0


def test_benchmark_competitiveness_missing_model_skipped():
    result = runner.invoke(
        app,
        [
            "benchmark",
            "benchmark-competitiveness",
            "--models",
            "mock-detect,no-such-model-xyz",
            "--max-images",
            "2",
            "--json",
        ],
    )
    # Should not crash; unknown models are skipped
    assert result.exit_code in (0, 1)


# ============================================================
# Phase 9 — debug-output CLI
# ============================================================


def _sample_image(tmp_path) -> str:
    img = Image.new("RGB", (320, 240), "blue")
    p = tmp_path / "test.jpg"
    img.save(str(p))
    return str(p)


def test_debug_output_json(tmp_path):
    img_path = _sample_image(tmp_path)
    result = runner.invoke(app, ["debug-output", "mock-detect", img_path, "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["model_id"] == "mock-detect"
    assert "total_detections" in payload
    assert "score_histogram" in payload
    assert "label_histogram_top10" in payload
    assert "invalid_boxes" in payload
    assert "image_size_wh" in payload
    assert payload["image_size_wh"] == [320, 240]


def test_debug_output_invalid_boxes_none_for_mock(tmp_path):
    """Mock engine returns clipped boxes; invalid_boxes should be 0."""
    img_path = _sample_image(tmp_path)
    result = runner.invoke(app, ["debug-output", "mock-detect", img_path, "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["invalid_boxes"] == 0


def test_debug_output_file_not_found():
    result = runner.invoke(app, ["debug-output", "mock-detect", "/no/such/file.jpg", "--json"])
    assert result.exit_code != 0


def test_debug_output_human_readable(tmp_path):
    img_path = _sample_image(tmp_path)
    result = runner.invoke(app, ["debug-output", "mock-detect", img_path])
    assert result.exit_code == 0
    assert "debug-output" in result.output.lower() or "Score histogram" in result.output


# ============================================================
# Phase 10 — recommend --goal
# ============================================================


def test_recommend_goal_accuracy_json():
    result = runner.invoke(app, ["recommend", "--task", "detect", "--goal", "accuracy", "--json"])
    assert result.exit_code == 0, result.output
    recs = json.loads(result.output)
    assert len(recs) > 0
    # Top result should prefer accuracy_grade
    top = recs[0]
    assert top["model_id"] != "dfine-n", "dfine-n (demo_fast) should not be top for accuracy goal"
    assert top["model_id"] != "rfdetr-nano", (
        "rfdetr-nano (demo_fast) should not be top for accuracy goal"
    )


def test_recommend_goal_fastest_demo_json():
    result = runner.invoke(
        app, ["recommend", "--task", "detect", "--goal", "fastest_demo", "--json"]
    )
    assert result.exit_code == 0
    recs = json.loads(result.output)
    top = recs[0]
    assert top["model_id"] in {"dfine-n", "dfine-n-coco", "rfdetr-nano", "mock-detect"}, (
        f"fastest_demo should prefer demo_fast models, got {top['model_id']}"
    )


def test_recommend_goal_best_segmentation_json():
    result = runner.invoke(app, ["recommend", "--goal", "best_segmentation", "--json"])
    assert result.exit_code == 0
    recs = json.loads(result.output)
    assert all(r["task"] in {"segment", "grounded_segment", "foundation_segment"} for r in recs), (
        "best_segmentation should return segmentation tasks"
    )


def test_recommend_goal_best_open_vocab_json():
    result = runner.invoke(app, ["recommend", "--goal", "best_open_vocab", "--json"])
    assert result.exit_code == 0
    recs = json.loads(result.output)
    assert all(r["task"] == "open_vocab_detect" for r in recs)


def test_recommend_experimental_sota_penalized_without_goal():
    """experimental_sota models should not surface in default recommendations."""
    result = runner.invoke(app, ["recommend", "--task", "detect", "--json"])
    assert result.exit_code == 0
    recs = json.loads(result.output)
    # Default top 5 should not include deim/rtdetrv4
    top_ids = {r["model_id"] for r in recs}
    experimental = {"deim-s", "deim-m", "deimv2-s", "deimv2-m", "rtdetrv4-s"}
    assert not (top_ids & experimental), (
        f"experimental_sota models surfaced without accuracy goal: {top_ids & experimental}"
    )


def test_recommend_unavailable_penalized():
    """unavailable_with_reason models should never appear in top recommendations."""
    result = runner.invoke(app, ["recommend", "--task", "detect", "--json"])
    assert result.exit_code == 0
    recs = json.loads(result.output)
    for r in recs:
        entry = default_registry().get(r["model_id"])
        assert entry.model_category != "unavailable_with_reason", (
            f"unavailable model {r['model_id']} surfaced in recommendations"
        )


# ============================================================
# Regression: all pre-v1.2.0 model IDs still resolve
# ============================================================


@pytest.mark.parametrize(
    "mid",
    [
        "mock-detect",
        "mock-classify",
        "mock-segment",
        "dfine-n",
        "dfine-s",
        "dfine-m",
        "dfine-l",
        "dfine-x",
        "rfdetr-nano",
        "rfdetr-small",
        "rfdetr-base",
        "rfdetr-medium",
        "rfdetr-large",
        "rfdetr-seg-nano",
        "rfdetr-seg-small",
        "rfdetr-seg-medium",
        "grounding-dino-tiny",
        "grounding-dino-swin-b",
        "swinv2-tiny",
        "swinv2-base",
        "sam-vit-base",
        "sam2-hiera-tiny",
        "grounded-sam",
        "grounded-sam2",
        "oneformer-swin-large",
        "co-dino-inst-vit-l-coco",
        "rtmpose-s",
    ],
)
def test_preexisting_model_ids_still_resolve(mid):
    e = default_registry().get(mid)
    assert e.id == mid
    assert e.license
    assert e.upstream_url
    assert e.model_category is not None


# ============================================================
# Registry integrity checks
# ============================================================


def test_no_accuracy_grade_model_without_implementation():
    """accuracy_grade models must be wired (real inference possible)."""
    for e in default_registry().list():
        if e.model_category == "accuracy_grade":
            assert e.implementation_status == "wired", (
                f"{e.id} is accuracy_grade but implementation_status={e.implementation_status!r}. "
                "Only wired models may be labelled accuracy_grade."
            )


def test_experimental_sota_never_auto_download():
    """experimental_sota models must not auto-download (not verified yet)."""
    for e in default_registry().list():
        if e.model_category == "experimental_sota":
            assert e.auto_download is False, (
                f"{e.id} is experimental_sota but auto_download=true. "
                "Experimental models must not auto-download."
            )


def test_demo_fast_models_are_wired():
    """demo_fast models should be wired (they must actually run)."""
    for e in default_registry().list():
        if e.model_category == "demo_fast":
            assert e.implementation_status == "wired", (
                f"{e.id} is demo_fast but implementation_status={e.implementation_status!r}."
            )


def test_total_model_count():
    """Sanity check: registry must have grown from v1.1.0 baseline."""
    assert len(list(default_registry().list())) >= 60, "Fewer models than expected in registry"
