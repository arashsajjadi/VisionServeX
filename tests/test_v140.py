# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tests for v1.4.0: output normalizer, Ultralytics-like ergonomics, model lifecycle CLI,
training/export capabilities, task aliases, video stubs."""

from __future__ import annotations

import json
import warnings

import pytest
from typer.testing import CliRunner

from visionservex.cli.main import app

runner = CliRunner()


# ============================================================
# Phase 1 — Output normalizer
# ============================================================


def test_normalize_list_xyxy():
    from visionservex.core.normalizer import normalize_detection

    det = normalize_detection({"xyxy": [10.0, 20.0, 100.0, 200.0], "score": 0.9, "label": "cat"})
    assert det.box.x1 == 10.0
    assert det.box.y1 == 20.0
    assert det.box.x2 == 100.0
    assert det.box.y2 == 200.0
    assert det.score == 0.9
    assert det.label == "cat"


def test_normalize_list_box():
    from visionservex.core.normalizer import normalize_detection

    det = normalize_detection(
        {"box": [10.0, 20.0, 100.0, 200.0], "confidence": 0.8, "category": "dog"}
    )
    assert det.box.x1 == 10.0
    assert abs(det.score - 0.8) < 1e-6
    assert det.label == "dog"


def test_normalize_list_bbox():
    from visionservex.core.normalizer import normalize_detection

    det = normalize_detection(
        {"bbox": [10.0, 20.0, 100.0, 200.0], "conf": 0.7, "class_name": "bird"}
    )
    assert det.box.x1 == 10.0
    assert abs(det.score - 0.7) < 1e-6
    assert det.label == "bird"


def test_normalize_bbox_xywh():
    from visionservex.core.normalizer import normalize_detection

    det = normalize_detection(
        {"bbox": [10.0, 20.0, 90.0, 80.0], "bbox_format": "xywh", "score": 0.5, "label": "car"}
    )
    assert det.box.x1 == 10.0
    assert det.box.y1 == 20.0
    assert abs(det.box.x2 - 100.0) < 1e-5  # x + w = 10 + 90
    assert abs(det.box.y2 - 100.0) < 1e-5  # y + h = 20 + 80


def test_normalize_dict_box_x1y1():
    from visionservex.core.normalizer import normalize_detection

    det = normalize_detection(
        {
            "box": {"x1": 10.0, "y1": 20.0, "x2": 100.0, "y2": 200.0},
            "score": 0.95,
            "label": "person",
        }
    )
    assert det.box.x1 == 10.0
    assert det.label == "person"


def test_normalize_dict_box_xmin_ymin():
    from visionservex.core.normalizer import normalize_detection

    det = normalize_detection(
        {
            "box": {"xmin": 5.0, "ymin": 10.0, "xmax": 50.0, "ymax": 100.0},
            "prob": 0.6,
            "class_name": "bicycle",
        }
    )
    assert det.box.x1 == 5.0
    assert det.box.xmax if hasattr(det.box, "xmax") else det.box.x2 == 50.0
    assert abs(det.score - 0.6) < 1e-6


def test_normalize_dict_xyxy_dict():
    from visionservex.core.normalizer import normalize_detection

    det = normalize_detection(
        {
            "xyxy": {"x1": 1.0, "y1": 2.0, "x2": 10.0, "y2": 20.0},
            "score": 0.4,
            "category_id": 0,
        }
    )
    assert det.box.x1 == 1.0
    assert det.class_id == 0
    assert det.label == "person"


def test_normalize_coordinates_dict():
    from visionservex.core.normalizer import normalize_detection

    det = normalize_detection(
        {
            "coordinates": {"left": 5.0, "top": 10.0, "right": 50.0, "bottom": 100.0},
            "score": 0.7,
            "name": "truck",
        }
    )
    assert det.box.x1 == 5.0


def test_normalize_score_aliases():
    from visionservex.core.normalizer import normalize_detection

    for key in ("score", "confidence", "conf", "probability", "prob"):
        det = normalize_detection({"xyxy": [0, 0, 1, 1], key: 0.77, "label": "x"})
        assert abs(det.score - 0.77) < 1e-6, f"score not found via key={key}"


def test_normalize_label_aliases():
    from visionservex.core.normalizer import normalize_detection

    for key in ("class_name", "label", "category", "name", "phrase"):
        det = normalize_detection({"xyxy": [0, 0, 1, 1], "score": 0.5, key: "myclass"})
        assert det.label == "myclass", f"label not found via key={key}"


def test_normalize_coco_official_id_mapping():
    from visionservex.core.normalizer import normalize_detection

    # COCO official ID 1 → contiguous 0 = "person"
    det = normalize_detection({"xyxy": [0, 0, 1, 1], "score": 0.5, "category_id": 1})
    assert det.class_id == 0
    assert det.label == "person"

    # COCO official ID 3 → contiguous 2 = "car"
    det = normalize_detection({"xyxy": [0, 0, 1, 1], "score": 0.5, "category_id": 3})
    assert det.class_id == 2
    assert det.label == "car"


def test_normalize_contiguous_id_passthrough():
    from visionservex.core.normalizer import normalize_detection

    # ID 0 is already contiguous → person
    det = normalize_detection({"xyxy": [0, 0, 1, 1], "score": 0.5, "class_id": 0})
    assert det.label == "person"


def test_normalize_unknown_schema_raises():
    from visionservex.core.normalizer import NormalizerError, normalize_detection

    with pytest.raises(NormalizerError) as exc_info:
        normalize_detection({"unknown_key": "value", "score": 0.9, "label": "x"})
    assert "OUTPUT_SCHEMA_UNRECOGNIZED" in str(exc_info.value)


def test_normalize_detections_list():
    from visionservex.core.normalizer import normalize_detections

    raws = [
        {"xyxy": [0, 0, 10, 10], "score": 0.9, "label": "cat"},
        {"box": [0, 0, 10, 10], "conf": 0.8, "label": "dog"},
    ]
    dets = normalize_detections(raws)
    assert len(dets) == 2


def test_normalize_detections_drops_warning():
    from visionservex.core.normalizer import AllPredictionsDroppedWarning, normalize_detections

    raws = [{"garbage": "data"}, {"also_garbage": 123}]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dets = normalize_detections(raws, warn_on_empty=True)
        assert len(dets) == 0
        assert any(issubclass(warning.category, AllPredictionsDroppedWarning) for warning in w)


def test_parse_api_response_dict_box():
    from visionservex.core.normalizer import parse_api_response

    # Simulates VisionServeX HTTP API response (box as dict)
    response = {
        "kind": "detection",
        "model_id": "dfine-s",
        "detections": [
            {
                "box": {"x1": 10.0, "y1": 20.0, "x2": 100.0, "y2": 200.0},
                "score": 0.9,
                "label": "cat",
            },
        ],
    }
    dets = parse_api_response(response)
    assert len(dets) == 1
    assert dets[0].label == "cat"
    assert dets[0].box.x1 == 10.0


def test_parse_api_response_non_detection():
    from visionservex.core.normalizer import parse_api_response

    # Classification response — should return empty list
    response = {"kind": "classification", "top_k": [["cat", 0.9]]}
    dets = parse_api_response(response)
    assert dets == []


def test_parse_api_response_empty():
    from visionservex.core.normalizer import parse_api_response

    assert parse_api_response({"kind": "detection", "detections": []}) == []


# ============================================================
# Phase 13A/B — VisionModel Ultralytics-like API
# ============================================================


def test_visionmodel_from_pretrained():
    from visionservex import VisionModel

    model = VisionModel.from_pretrained("mock-detect")
    assert model.entry.id == "mock-detect"


def test_visionmodel_from_registry():
    from visionservex import VisionModel

    model = VisionModel.from_registry("mock-detect")
    assert model.entry.id == "mock-detect"


def test_visionmodel_from_checkpoint_raises():
    from visionservex import VisionModel

    with pytest.raises(NotImplementedError) as exc_info:
        VisionModel.from_checkpoint("some/path.pt")
    assert "CHECKPOINT_LOAD_UNSUPPORTED" in str(exc_info.value)


def test_visionmodel_to_device():
    from visionservex import VisionModel

    model = VisionModel("mock-detect")
    returned = model.to("cpu")
    assert returned is model
    assert model.device == "cpu"


def test_visionmodel_names():
    from visionservex import VisionModel
    from visionservex.core.normalizer import COCO80_NAMES

    model = VisionModel("mock-detect")
    names = model.names
    assert isinstance(names, list)
    assert len(names) == 80
    assert "person" in names
    assert names == COCO80_NAMES


def test_visionmodel_supports_predict():
    from visionservex import VisionModel

    model = VisionModel("mock-detect")
    result = model.supports("predict")
    assert result["supported"] is True


def test_visionmodel_supports_train_not():
    from visionservex import VisionModel

    model = VisionModel("mock-detect")
    result = model.supports("train")
    assert result["supported"] is False


def test_visionmodel_supports_video_not():
    from visionservex import VisionModel

    model = VisionModel("mock-detect")
    result = model.supports("video")
    assert result["supported"] is False
    assert (
        "NOT_IMPLEMENTED" in result.get("reason", "").upper()
        or "roadmap" in result.get("hint", "").lower()
    )


def test_visionmodel_training_info():
    from visionservex import VisionModel

    model = VisionModel("mock-detect")
    info = model.training_info()
    assert "train_supported" in info
    assert "finetune_supported" in info
    assert isinstance(info["train_supported"], bool)


def test_visionmodel_export_info():
    from visionservex import VisionModel

    model = VisionModel("mock-detect")
    info = model.export_info()
    assert "model_id" in info
    assert "family" in info


def test_visionmodel_cache_info():
    from visionservex import VisionModel

    model = VisionModel("mock-detect")
    info = model.cache_info()
    assert "model_id" in info
    assert "cached" in info
    assert "cache_dir" in info


def test_visionmodel_checkpoint_info():
    from visionservex import VisionModel

    model = VisionModel("mock-detect")
    info = model.checkpoint_info()
    assert "model_id" in info
    assert "checkpoint_trust_level" in info
    assert "license" in info
    assert "verified_by_visionservex" in info


def test_visionmodel_val_detect_no_dataset():
    from visionservex import VisionModel

    model = VisionModel("mock-detect")
    result = model.val()
    assert result["status"] == "DATASET_REQUIRED"


def test_visionmodel_val_non_detect():
    from visionservex import VisionModel

    model = VisionModel("mock-classify")
    result = model.val(dataset="yolo:/tmp/nothing")
    assert result["status"] == "BENCHMARK_NOT_IMPLEMENTED"


def test_visionmodel_pull_synthetic_returns_none():
    from visionservex import VisionModel

    model = VisionModel("mock-detect")
    path = model.pull()
    assert path is None  # synthetic models don't download


# ============================================================
# Phase 13E — Results enhancements
# ============================================================


def test_result_to_csv(tmp_path):
    from visionservex import VisionModel

    model = VisionModel("mock-detect")
    model._ensure_loaded()
    from PIL import Image

    img = Image.new("RGB", (320, 240))
    result = model.predict(img)
    csv = result.to_csv()
    assert isinstance(csv, str)
    lines = csv.strip().splitlines()
    assert len(lines) >= 1
    assert "model_id" in lines[0].lower() or "label" in lines[0].lower()


def test_result_debug():
    from visionservex import VisionModel

    model = VisionModel("mock-detect")
    model._ensure_loaded()
    from PIL import Image

    img = Image.new("RGB", (320, 240))
    result = model.predict(img)
    dbg = result.debug()
    assert isinstance(dbg, str)
    assert "mock-detect" in dbg
    assert "Latency" in dbg


def test_result_to_json_parses():
    from visionservex import VisionModel

    model = VisionModel("mock-detect")
    model._ensure_loaded()
    from PIL import Image

    img = Image.new("RGB", (320, 240))
    result = model.predict(img)
    payload = json.loads(result.to_json())
    assert "detections" in payload


# ============================================================
# Phase 13D — Model lifecycle CLI
# ============================================================


def test_model_info_cli_json():
    result = runner.invoke(app, ["model", "info", "mock-detect", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["id"] == "mock-detect"
    assert "cached" in data


def test_model_info_cli_not_found():
    result = runner.invoke(app, ["model", "info", "no-such-model-xyz", "--json"])
    assert result.exit_code != 0
    data = json.loads(result.output)
    assert "error" in data


def test_model_pull_dry_run_json():
    result = runner.invoke(app, ["model", "pull", "mock-detect", "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["dry_run"] is True
    assert data["model_id"] == "mock-detect"


def test_model_checkpoint_info_json():
    result = runner.invoke(app, ["model", "checkpoint-info", "mock-detect", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["model_id"] == "mock-detect"
    assert "checkpoint_trust_level" in data
    assert "verified_by_visionservex" in data


def test_model_cache_json():
    result = runner.invoke(app, ["model", "cache", "mock-detect", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "model_id" in data or isinstance(data, list)


def test_model_list_local_json():
    result = runner.invoke(app, ["model", "list-local", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list)


# ============================================================
# Training capabilities CLI
# ============================================================


def test_training_capabilities_all_json():
    result = runner.invoke(app, ["training", "capabilities", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) > 0
    for item in data:
        assert "train_supported" in item
        assert "finetune_supported" in item


def test_training_capabilities_model_json():
    result = runner.invoke(app, ["training", "capabilities", "--model", "mock-detect", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "model_id" in data
    assert isinstance(data["train_supported"], bool)


def test_train_cmd_not_supported():
    result = runner.invoke(app, ["train", "mock-detect", "--data", "data.yaml", "--json"])
    # Most models return TRAINING_NOT_SUPPORTED (exit 2)
    assert result.exit_code in (1, 2)
    data = json.loads(result.output)
    assert "TRAINING_NOT_SUPPORTED" in data.get("status", "")


def test_finetune_cmd_not_supported():
    result = runner.invoke(app, ["finetune", "mock-detect", "--data", "data.yaml", "--json"])
    assert result.exit_code in (1, 2)
    data = json.loads(result.output)
    assert "TRAINING_NOT_SUPPORTED" in data.get("status", "")


# ============================================================
# Export capabilities CLI
# ============================================================


def test_export_capabilities_json():
    result = runner.invoke(app, ["export-cmd", "capabilities", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)


def test_export_capabilities_model_json():
    result = runner.invoke(app, ["export-cmd", "capabilities", "--model", "mock-detect", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "model_id" in data


def test_export_unsupported():
    result = runner.invoke(
        app,
        [
            "export-cmd",
            "export",
            "mock-detect",
            "--format",
            "tensorrt",
            "--out",
            "/tmp/test.trt",
            "--json",
        ],
    )
    assert result.exit_code == 2
    data = json.loads(result.output)
    assert "EXPORT_UNSUPPORTED" in data.get("status", "")


# ============================================================
# CLI task aliases
# ============================================================


def test_detect_alias(tmp_path):
    img = __import__("PIL").Image.new("RGB", (320, 240), "blue")
    img_path = tmp_path / "test.jpg"
    img.save(str(img_path))
    result = runner.invoke(app, ["detect", "mock-detect", str(img_path), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "detections" in data


def test_segment_alias(tmp_path):
    img = __import__("PIL").Image.new("RGB", (320, 240), "green")
    img_path = tmp_path / "test.jpg"
    img.save(str(img_path))
    result = runner.invoke(app, ["segment", "mock-segment", str(img_path), "--json"])
    assert result.exit_code == 0, result.output


def test_classify_alias(tmp_path):
    img = __import__("PIL").Image.new("RGB", (320, 240), "red")
    img_path = tmp_path / "test.jpg"
    img.save(str(img_path))
    result = runner.invoke(app, ["classify", "mock-classify", str(img_path), "--json"])
    assert result.exit_code == 0, result.output


def test_open_vocab_alias(tmp_path):
    img = __import__("PIL").Image.new("RGB", (320, 240), "yellow")
    img_path = tmp_path / "test.jpg"
    img.save(str(img_path))
    result = runner.invoke(
        app, ["open-vocab", "mock-open-vocab", str(img_path), "--prompt", "cat,dog", "--json"]
    )
    assert result.exit_code == 0, result.output


def test_val_missing_dataset():
    result = runner.invoke(
        app, ["val", "mock-detect", "--dataset", "yolo:/nonexistent/path", "--json"]
    )
    # Should fail because path doesn't exist
    assert result.exit_code != 0 or json.loads(result.output).get("status") in (
        "DATASET_PARSE_FAILED",
        "VAL_FAILED",
        "DATASET_REQUIRED",
    )


# ============================================================
# Video stubs
# ============================================================


@pytest.mark.parametrize(
    "subcmd,args",
    [
        ("predict", ["mock-detect", "video.mp4"]),
        ("track", ["mock-detect", "video.mp4"]),
        ("stream", ["mock-detect", "--source", "webcam"]),
    ],
)
def test_video_stubs_exit2(subcmd, args):
    result = runner.invoke(app, ["video", subcmd, *args, "--json"])
    assert result.exit_code == 2
    data = json.loads(result.output)
    assert "NOT_IMPLEMENTED" in data["status"]
    assert "roadmap" in data


# ============================================================
# Version check
# ============================================================


def test_version_is_at_least_140():
    from visionservex import __version__

    major, minor, _ = (int(x) for x in __version__.split("."))
    assert (major, minor) >= (1, 4), f"Expected at least 1.4.x, got {__version__}"


# ============================================================
# Normalizer exports at top level
# ============================================================


def test_top_level_normalizer_exports():
    from visionservex import normalize_detection, normalize_detections, parse_api_response

    det = normalize_detection({"xyxy": [0, 0, 10, 10], "score": 0.5, "label": "test"})
    assert det.label == "test"

    dets = normalize_detections([{"xyxy": [0, 0, 10, 10], "score": 0.5, "label": "x"}])
    assert len(dets) == 1

    result = parse_api_response({"kind": "detection", "detections": []})
    assert result == []
