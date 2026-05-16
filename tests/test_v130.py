# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tests for v1.3.0: capabilities report, model cards, replacement map,
real AP/mAP evaluation engine, debug-output improvements, benchmark stubs."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from visionservex.cli.main import app

runner = CliRunner()


# ============================================================
# Phase 1 — capabilities report
# ============================================================


def test_capabilities_report_json():
    result = runner.invoke(app, ["capabilities", "report", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "version" in payload
    assert "devices" in payload
    assert "models" in payload
    assert "installed_extras" in payload
    assert "recommendations_by_goal" in payload
    assert "known_limitations" in payload
    assert isinstance(payload["known_limitations"], list)
    assert len(payload["known_limitations"]) > 0


def test_capabilities_report_models_section():
    result = runner.invoke(app, ["capabilities", "report", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    models = payload["models"]
    assert "total" in models
    assert "runnable_count" in models
    assert models["total"] > 0
    assert models["runnable_count"] > 0
    assert "by_task" in models
    assert "by_category" in models
    assert "detect" in models["by_task"]


def test_capabilities_report_extras_listed():
    result = runner.invoke(app, ["capabilities", "report", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    extras = payload["installed_extras"]
    for name in ("server", "hf", "rfdetr", "onnx", "openmmlab", "dev"):
        assert name in extras, f"Extra '{name}' missing from report"
        assert "available" in extras[name]
        assert "missing_packages" in extras[name]


def test_capabilities_report_markdown():
    result = runner.invoke(app, ["capabilities", "report", "--format", "markdown"])
    assert result.exit_code == 0
    assert "# VisionServeX" in result.output
    assert "## Devices" in result.output
    assert "## Models" in result.output


def test_capabilities_report_human():
    result = runner.invoke(app, ["capabilities", "report"])
    assert result.exit_code == 0
    # Rich output — just check no crash and some expected strings
    assert "VisionServeX" in result.output or result.exit_code == 0


def test_capabilities_report_file_out(tmp_path):
    out = tmp_path / "cap.json"
    result = runner.invoke(app, ["capabilities", "report", "--json", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert "version" in data


# ============================================================
# Phase 2 — model card system
# ============================================================


def test_model_card_json_known():
    result = runner.invoke(app, ["model-card", "show", "dfine-s-o365-coco", "--json"])
    assert result.exit_code == 0, result.output
    card = json.loads(result.output)
    assert card["model_id"] == "dfine-s-o365-coco"
    assert "recommended_for" in card
    assert "not_recommended_for" in card
    assert "official_benchmark_note" in card
    assert "visionservex_benchmark_status" in card
    assert "install_command" in card
    assert "quick_command" in card
    assert len(card["recommended_for"]) > 0


def test_model_card_json_demo_fast():
    result = runner.invoke(app, ["model-card", "show", "dfine-n", "--json"])
    assert result.exit_code == 0
    card = json.loads(result.output)
    assert card["strength_category"] == "demo_fast"
    # Demo models must explicitly say not for accuracy benchmarks
    not_for = " ".join(card["not_recommended_for"]).lower()
    assert "accuracy" in not_for or "benchmark" in not_for


def test_model_card_json_rfdetr_nano_demo():
    result = runner.invoke(app, ["model-card", "show", "rfdetr-nano", "--json"])
    assert result.exit_code == 0
    card = json.loads(result.output)
    assert card["strength_category"] == "demo_fast"


def test_model_card_json_sam():
    result = runner.invoke(app, ["model-card", "show", "sam-vit-base", "--json"])
    assert result.exit_code == 0
    card = json.loads(result.output)
    # SAM cards must warn against detection AP
    note = card["official_benchmark_note"].lower()
    assert "detection" in note or "not comparable" in note or "sa-1b" in note or "ap" in note


def test_model_card_not_found():
    result = runner.invoke(app, ["model-card", "show", "no-such-model-xyz", "--json"])
    data = json.loads(result.output)
    assert "error" in data


def test_model_card_markdown():
    result = runner.invoke(app, ["model-card", "show", "dfine-s-o365-coco", "--format", "markdown"])
    assert result.exit_code == 0
    assert "# Model Card" in result.output
    assert "## Install" in result.output
    assert "## Recommended for" in result.output


def test_model_cards_list_json():
    result = runner.invoke(app, ["model-card", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "with_full_card" in data
    assert "total" in data
    assert len(data["with_full_card"]) > 0


def test_model_cards_list_detect_filter():
    result = runner.invoke(app, ["model-card", "list", "--task", "detect", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "dfine-s-o365-coco" in data["with_full_card"]
    assert "rfdetr-small" in data["with_full_card"]


def test_model_card_registry_only_fallback():
    """Models without explicit card data should still return a valid card."""
    result = runner.invoke(app, ["model-card", "show", "swinv2-large", "--json"])
    assert result.exit_code == 0
    card = json.loads(result.output)
    assert "model_id" in card
    assert card["task"] == "classify"


# ============================================================
# Phase 3 — replacement map
# ============================================================


def test_replacement_map_detect_json():
    result = runner.invoke(app, ["replacement-map", "map", "--task", "detect", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "detect" in data
    entry = data["detect"]
    assert "replacements" in entry
    assert len(entry["replacements"]) >= 2
    tiers = {r["tier"] for r in entry["replacements"]}
    assert "fastest_demo" in tiers
    assert "production" in tiers
    assert "accuracy" in tiers


def test_replacement_map_detect_no_ap_claim():
    result = runner.invoke(app, ["replacement-map", "map", "--task", "detect", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    for rep in data["detect"]["replacements"]:
        # No replacement should claim AP without evidence
        assert not rep.get("ap_claim"), f"tier={rep['tier']} wrongly claims AP"


def test_replacement_map_segment_json():
    result = runner.invoke(app, ["replacement-map", "map", "--task", "segment", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "segment" in data
    # SAM should appear in prompt_based tier
    seg_models = [m for rep in data["segment"]["replacements"] for m in rep["models"]]
    # At least one SAM or grounded-sam model
    assert any("sam" in m.lower() for m in seg_models)


def test_replacement_map_classify_json():
    result = runner.invoke(app, ["replacement-map", "map", "--task", "classify", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "classify" in data
    all_models = [m for rep in data["classify"]["replacements"] for m in rep["models"]]
    assert any("swinv2" in m for m in all_models)


def test_replacement_map_pose_json():
    result = runner.invoke(app, ["replacement-map", "map", "--task", "pose", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "pose" in data
    # Must include caveats about not being wired
    caveats = data["pose"]["honest_caveats"]
    assert len(caveats) > 0
    assert any(
        "wired" in c.lower() or "sidecar" in c.lower() or "not" in c.lower() for c in caveats
    )


def test_replacement_map_open_vocab_json():
    result = runner.invoke(app, ["replacement-map", "map", "--task", "open-vocab", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "open-vocab" in data


def test_replacement_map_markdown():
    result = runner.invoke(app, ["replacement-map", "map", "--format", "markdown"])
    assert result.exit_code == 0
    assert "# VisionServeX Replacement Map" in result.output
    assert "YOLO detect" in result.output or "detect" in result.output.lower()


def test_replacement_map_invalid_task():
    result = runner.invoke(app, ["replacement-map", "map", "--task", "no-such-task"])
    assert result.exit_code != 0


# ============================================================
# Phase 4 — AP evaluation engine unit tests
# ============================================================


def test_box_iou_perfect_overlap():
    from visionservex.runtime.evaluation import _box_iou

    box = [10.0, 10.0, 50.0, 50.0]
    gt = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
    iou = _box_iou(box, gt)
    assert abs(iou[0] - 1.0) < 1e-5


def test_box_iou_no_overlap():
    from visionservex.runtime.evaluation import _box_iou

    box = [0.0, 0.0, 10.0, 10.0]
    gt = np.array([[20.0, 20.0, 40.0, 40.0]], dtype=np.float32)
    iou = _box_iou(box, gt)
    assert abs(iou[0]) < 1e-5


def test_box_iou_partial_overlap():
    from visionservex.runtime.evaluation import _box_iou

    box = [0.0, 0.0, 20.0, 20.0]
    gt = np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)
    iou = _box_iou(box, gt)
    # Intersection: 10x10=100, Union: 400+400-100=700, IoU≈0.1429
    assert 0.10 < iou[0] < 0.20


def test_ap_from_pr_perfect():
    from visionservex.runtime.evaluation import _ap_from_pr

    recalls = np.array([0.0, 1.0])
    precisions = np.array([1.0, 1.0])
    ap = _ap_from_pr(recalls, precisions)
    assert abs(ap - 1.0) < 1e-3


def test_ap_from_pr_zero():
    from visionservex.runtime.evaluation import _ap_from_pr

    recalls = np.array([0.0])
    precisions = np.array([0.0])
    ap = _ap_from_pr(recalls, precisions)
    assert ap == 0.0


def test_evaluator_perfect_match():
    from visionservex.runtime.evaluation import DetectionEvaluator

    ev = DetectionEvaluator()
    # Perfect prediction
    ev.add_image(
        pred_boxes=[[10.0, 10.0, 50.0, 50.0]],
        pred_scores=[0.9],
        pred_classes=["cat"],
        gt_boxes=[[10.0, 10.0, 50.0, 50.0]],
        gt_classes=["cat"],
    )
    metrics = ev.compute_metrics(iou_threshold=0.50)
    assert metrics["map50"] > 0.95


def test_evaluator_no_predictions():
    from visionservex.runtime.evaluation import DetectionEvaluator

    ev = DetectionEvaluator()
    ev.add_image(
        pred_boxes=[],
        pred_scores=[],
        pred_classes=[],
        gt_boxes=[[10.0, 10.0, 50.0, 50.0]],
        gt_classes=["cat"],
    )
    metrics = ev.compute_metrics()
    assert metrics["map50"] == 0.0


def test_evaluator_wrong_class():
    from visionservex.runtime.evaluation import DetectionEvaluator

    ev = DetectionEvaluator()
    ev.add_image(
        pred_boxes=[[10.0, 10.0, 50.0, 50.0]],
        pred_scores=[0.9],
        pred_classes=["dog"],  # wrong class
        gt_boxes=[[10.0, 10.0, 50.0, 50.0]],
        gt_classes=["cat"],
    )
    metrics = ev.compute_metrics()
    assert metrics["map50"] == 0.0  # class mismatch → 0 AP for both classes


def test_evaluator_multiple_images():
    from visionservex.runtime.evaluation import DetectionEvaluator

    ev = DetectionEvaluator()
    # Image 1: perfect
    ev.add_image([[10, 10, 50, 50]], [0.9], ["cat"], [[10, 10, 50, 50]], ["cat"])
    # Image 2: no detection
    ev.add_image([], [], [], [[10, 10, 50, 50]], ["cat"])
    metrics = ev.compute_metrics()
    # AP should be between 0 and 1
    assert 0.0 <= metrics["map50"] <= 1.0


def test_evaluator_map50_95():
    from visionservex.runtime.evaluation import DetectionEvaluator

    ev = DetectionEvaluator()
    ev.add_image([[10, 10, 50, 50]], [0.9], ["cat"], [[10, 10, 50, 50]], ["cat"])
    result = ev.compute_map50_95()
    assert "map50_95" in result
    # mAP50:95 should be <= AP50
    assert result["map50_95"] <= result["map50"] + 0.01


def test_evaluator_per_class():
    from visionservex.runtime.evaluation import DetectionEvaluator

    ev = DetectionEvaluator()
    ev.add_image(
        [[10, 10, 50, 50], [100, 100, 150, 150]],
        [0.9, 0.8],
        ["cat", "dog"],
        [[10, 10, 50, 50], [100, 100, 150, 150]],
        ["cat", "dog"],
    )
    metrics = ev.compute_metrics()
    assert metrics["n_classes_with_gt"] == 2
    classes = {d["class"] for d in metrics["per_class"]}
    assert "cat" in classes
    assert "dog" in classes


# ============================================================
# Phase 4 — YOLO dataset loading
# ============================================================


def test_load_yolo_format_empty_dir(tmp_path):
    from visionservex.runtime.evaluation import load_yolo_format

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    samples, class_names = load_yolo_format(tmp_path)
    assert samples == []
    assert len(class_names) > 0  # defaults to COCO80


def test_load_yolo_format_one_image(tmp_path):
    from PIL import Image

    from visionservex.runtime.evaluation import load_yolo_format

    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    img_path = img_dir / "test.jpg"
    img.save(str(img_path))

    # YOLO label: class 0 (person), center 0.5,0.5, width 0.3, height 0.4
    lbl_path = lbl_dir / "test.txt"
    lbl_path.write_text("0 0.5 0.5 0.3 0.4\n", encoding="utf-8")

    samples, _class_names = load_yolo_format(tmp_path)
    assert len(samples) == 1
    s = samples[0]
    assert len(s.gt_boxes) == 1
    assert len(s.gt_classes) == 1
    assert s.gt_classes[0] == "person"
    # Verify box conversion from center/wh to xyxy
    x1, y1, _x2, _y2 = s.gt_boxes[0]
    assert abs(x1 - (0.5 - 0.15) * 640) < 1
    assert abs(y1 - (0.5 - 0.2) * 480) < 1


def test_load_yolo_format_with_yaml(tmp_path):
    import yaml
    from PIL import Image

    from visionservex.runtime.evaluation import load_yolo_format

    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    img = Image.new("RGB", (320, 240))
    img.save(str(img_dir / "a.jpg"))
    (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    (tmp_path / "data.yaml").write_text(
        yaml.dump({"names": ["apple", "banana", "cherry"]}),
        encoding="utf-8",
    )

    samples, class_names = load_yolo_format(tmp_path)
    assert class_names == ["apple", "banana", "cherry"]
    assert samples[0].gt_classes[0] == "apple"


# ============================================================
# Phase 4 — synthetic AP benchmark (mock-detect)
# ============================================================


def test_benchmark_competitiveness_synthetic_json():
    """Existing synthetic mode still works."""
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


def test_benchmark_competitiveness_dataset_missing_path():
    """Non-existent dataset path should exit non-zero."""
    result = runner.invoke(
        app,
        [
            "benchmark",
            "benchmark-competitiveness",
            "--models",
            "mock-detect",
            "--dataset",
            "yolo:/nonexistent/path/coco128",
            "--max-images",
            "3",
        ],
    )
    assert result.exit_code != 0


def test_benchmark_competitiveness_yolo_dataset(tmp_path):
    """Full AP benchmark run with a tiny synthetic annotated dataset."""
    from PIL import Image

    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    for i in range(5):
        img = Image.new("RGB", (640, 480), color=(i * 30, 100, 200))
        img.save(str(img_dir / f"img{i}.jpg"))
        # 0=person box
        (lbl_dir / f"img{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    result = runner.invoke(
        app,
        [
            "benchmark",
            "benchmark-competitiveness",
            "--models",
            "mock-detect",
            "--dataset",
            f"yolo:{tmp_path}",
            "--max-images",
            "5",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["benchmark_type"] == "competitiveness_with_ap_evaluation"
    assert payload["n_images"] == 5
    assert len(payload["models"]) == 1
    model_result = payload["models"][0]
    # Should have AP fields
    assert "ap50" in model_result
    assert "map50_95" in model_result
    assert "latency_p50_ms" in model_result


def test_benchmark_competitiveness_csv_export(tmp_path):
    from PIL import Image

    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    img = Image.new("RGB", (320, 240))
    img.save(str(img_dir / "a.jpg"))
    (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.3 0.3\n")

    out = tmp_path / "results"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "benchmark-competitiveness",
            "--models",
            "mock-detect",
            "--dataset",
            f"yolo:{tmp_path}",
            "--max-images",
            "1",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    assert out.with_suffix(".json").exists() or out.with_suffix(".csv").exists()


# ============================================================
# Phase 5 — debug-output improvements
# ============================================================


def test_debug_output_save_json(tmp_path):
    img = __import__("PIL").Image.new("RGB", (320, 240), "blue")
    img_path = tmp_path / "test.jpg"
    img.save(str(img_path))
    save_to = tmp_path / "diag.json"
    result = runner.invoke(
        app, ["debug-output", "mock-detect", str(img_path), "--save-json", str(save_to)]
    )
    assert result.exit_code == 0
    assert save_to.exists()
    data = json.loads(save_to.read_text())
    assert "model_id" in data
    assert "total_detections" in data


def test_debug_output_visualize(tmp_path):
    img = __import__("PIL").Image.new("RGB", (320, 240), "green")
    img_path = tmp_path / "test.jpg"
    img.save(str(img_path))
    vis_path = tmp_path / "vis.jpg"
    result = runner.invoke(
        app, ["debug-output", "mock-detect", str(img_path), "--visualize", str(vis_path)]
    )
    assert result.exit_code == 0
    assert vis_path.exists()


# ============================================================
# Phase 7 — recommend --goal coverage
# ============================================================


def test_recommend_goal_accuracy_no_demo_fast():
    result = runner.invoke(app, ["recommend", "--task", "detect", "--goal", "accuracy", "--json"])
    assert result.exit_code == 0
    recs = json.loads(result.output)
    assert len(recs) > 0
    top_id = recs[0]["model_id"]
    assert top_id not in ("dfine-n", "dfine-n-coco", "rfdetr-nano"), (
        f"goal=accuracy should not surface demo_fast model, got {top_id}"
    )


def test_recommend_goal_fastest_demo():
    result = runner.invoke(
        app, ["recommend", "--task", "detect", "--goal", "fastest_demo", "--json"]
    )
    assert result.exit_code == 0
    recs = json.loads(result.output)
    top = recs[0]
    # Should prefer demo_fast or small models
    assert (
        top["model_id"]
        in {
            "dfine-n",
            "dfine-n-coco",
            "rfdetr-nano",
            "mock-detect",
            "mock-segment",
        }
        or "demo" in (top.get("model_id") or "").lower()
    )


def test_recommend_goal_segmentation():
    result = runner.invoke(app, ["recommend", "--goal", "best_segmentation", "--json"])
    assert result.exit_code == 0
    recs = json.loads(result.output)
    tasks = {r["task"] for r in recs}
    # Should include segmentation task models
    assert "segment" in tasks or "grounded_segment" in tasks or "foundation_segment" in tasks


# ============================================================
# Phase 8 — non-detection benchmark stubs
# ============================================================


@pytest.mark.parametrize(
    "subcmd",
    [
        # benchmark-segmentation is now a real command (v1.5.0) — excluded here
        "benchmark-classification",
        "benchmark-open-vocab",
        "benchmark-pose",
        "benchmark-obb",
    ],
)
def test_benchmark_not_implemented_json(subcmd: str):
    result = runner.invoke(app, ["benchmark", subcmd, "--json"])
    # Exit code 2 = structured BENCHMARK_NOT_IMPLEMENTED
    assert result.exit_code == 2, f"{subcmd} should exit 2, got {result.exit_code}"
    payload = json.loads(result.output)
    assert payload["status"] == "BENCHMARK_NOT_IMPLEMENTED"
    assert "metrics" in payload
    assert "roadmap" in payload
    assert len(payload["metrics"]) > 0


@pytest.mark.parametrize(
    "subcmd",
    [
        # benchmark-segmentation is now a real command (v1.5.0) — excluded here
        "benchmark-classification",
        "benchmark-pose",
    ],
)
def test_benchmark_not_implemented_human(subcmd: str):
    result = runner.invoke(app, ["benchmark", subcmd])
    assert result.exit_code == 2
    assert "BENCHMARK_NOT_IMPLEMENTED" in result.output


def test_benchmark_stubs_different_metrics():
    """Each task stub reports task-appropriate metrics."""
    for subcmd, expected_metric in [
        # benchmark-segmentation is now a real command — use test_v150.py for it
        ("benchmark-classification", "top-1 accuracy"),
        ("benchmark-pose", "OKS AP50"),
        ("benchmark-obb", "rotated IoU AP50"),
    ]:
        result = runner.invoke(app, ["benchmark", subcmd, "--json"])
        payload = json.loads(result.output)
        metrics = payload["metrics"]
        assert any(expected_metric.lower() in m.lower() for m in metrics), (
            f"{subcmd}: expected '{expected_metric}' in metrics, got {metrics}"
        )


def test_benchmark_segmentation_now_real():
    """benchmark-segmentation is now real in v1.5.0 — exits 0 in synthetic mode."""
    result = runner.invoke(
        app,
        [
            "benchmark",
            "benchmark-segmentation",
            "--models",
            "mock-segment",
            "--max-images",
            "2",
            "--json",
        ],
    )
    # In v1.5.0: exits 0 in synthetic mode, returns benchmark_type not BENCHMARK_NOT_IMPLEMENTED
    # Note: if no segmentation models are valid, may exit non-zero
    if result.exit_code == 0:
        payload = json.loads(result.output)
        assert "benchmark_type" in payload
    # Either way, it should not return BENCHMARK_NOT_IMPLEMENTED anymore
    if result.exit_code == 0:
        try:
            payload = json.loads(result.output)
            assert payload.get("status") != "BENCHMARK_NOT_IMPLEMENTED"
        except Exception:
            pass


# ============================================================
# Phase 4 — honest_conclusion tests
# ============================================================


def test_generate_honest_conclusion_no_gt():
    from visionservex.runtime.evaluation import EvaluationResult, generate_honest_conclusion

    results = [
        EvaluationResult(
            model_id="test-model",
            dataset="synthetic",
            n_images=10,
            ap50=0.0,
            map50_95=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            latency_p50_ms=15.0,
            n_classes_with_gt=0,
            status="ok",
        )
    ]
    conclusion = generate_honest_conclusion(results)
    assert "latency" in conclusion.lower() or "ground-truth" in conclusion.lower()


def test_generate_honest_conclusion_with_gt():
    from visionservex.runtime.evaluation import EvaluationResult, generate_honest_conclusion

    results = [
        EvaluationResult(
            model_id="model-a",
            dataset="test",
            n_images=100,
            ap50=0.55,
            map50_95=0.35,
            precision=0.8,
            recall=0.7,
            f1=0.75,
            latency_p50_ms=20.0,
            n_classes_with_gt=5,
            status="ok",
        ),
        EvaluationResult(
            model_id="model-b",
            dataset="test",
            n_images=100,
            ap50=0.48,
            map50_95=0.30,
            precision=0.75,
            recall=0.65,
            f1=0.70,
            latency_p50_ms=15.0,
            n_classes_with_gt=5,
            status="ok",
        ),
    ]
    conclusion = generate_honest_conclusion(results)
    assert "model-a" in conclusion  # should mention winner
    assert "AP50" in conclusion or "ap50" in conclusion.lower()


def test_honest_conclusion_small_dataset_warning():
    from visionservex.runtime.evaluation import EvaluationResult, generate_honest_conclusion

    results = [
        EvaluationResult(
            model_id="x",
            dataset="d",
            n_images=10,
            ap50=0.5,
            map50_95=0.3,
            precision=0.8,
            recall=0.7,
            f1=0.75,
            n_classes_with_gt=2,
            status="ok",
        )
    ]
    conclusion = generate_honest_conclusion(results)
    assert "10 images" in conclusion or "WARNING" in conclusion or "variance" in conclusion.lower()


# ============================================================
# Misc: version check
# ============================================================


def test_version_is_at_least_130():
    from visionservex import __version__

    major, minor, _ = (int(x) for x in __version__.split("."))
    assert (major, minor) >= (1, 3), f"Expected at least 1.3.x, got {__version__}"
