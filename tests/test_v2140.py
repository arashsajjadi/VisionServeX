# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for v2.14.0 visualization / live / annotate infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner


def _make_image(path: Path, w: int = 320, h: int = 240) -> Path:
    img = Image.new("RGB", (w, h), color=(40, 40, 40))
    img.save(path)
    return path


def test_visualization_draw_detections(tmp_path: Path) -> None:
    from visionservex.visualization import draw_detections

    img = Image.new("RGB", (320, 240), color=(20, 20, 20))
    dets = [
        {"box": [10, 20, 100, 90], "score": 0.91, "class_name": "person"},
        {"box": {"x1": 120, "y1": 30, "x2": 200, "y2": 110}, "score": 0.55, "class_name": "bag"},
    ]
    out = draw_detections(img, dets)
    assert isinstance(out, Image.Image)
    assert out.size == (320, 240)


def test_visualization_annotate_image_routes_by_task(tmp_path: Path) -> None:
    from visionservex.visualization import annotate_image

    img = Image.new("RGB", (320, 240), color=(0, 0, 0))
    payload = {
        "task": "detect",
        "detections": [{"box": [5, 5, 60, 60], "score": 0.8, "class_name": "x"}],
    }
    out = annotate_image(img, payload)
    assert isinstance(out, Image.Image)


def test_video_io_dry_run_webcam() -> None:
    from visionservex.runtime.video_io import open_video_source

    s = open_video_source("0", dry_run=True)
    assert s.source == "0"


def test_video_io_folder_source(tmp_path: Path) -> None:
    from visionservex.runtime.video_io import open_video_source

    folder = tmp_path / "frames"
    folder.mkdir()
    for i in range(3):
        Image.new("RGB", (64, 48), color=(i * 30, 0, 0)).save(folder / f"frame_{i:03d}.png")

    s = open_video_source(str(folder), max_frames=2)
    frames = list(s.iter_frames())
    assert len(frames) == 2
    assert frames[0].image.size == (64, 48)


def test_live_dry_run_yields_dry_run_ok() -> None:
    from visionservex.runtime.live import LiveConfig, run_live

    cfg = LiveConfig(source="0", model_id="anything", task="detect", dry_run=True)
    payloads = list(run_live(cfg))
    assert len(payloads) == 1
    assert payloads[0].get("code") == "DRY_RUN_OK"


def test_cli_live_dry_run_json() -> None:
    from visionservex.cli.main import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "live",
            "--source",
            "0",
            "--model",
            "dfine-s-o365-coco",
            "--task",
            "detect",
            "--dry-run",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "DRY_RUN_OK"


def test_cli_draw_image(tmp_path: Path) -> None:
    from visionservex.cli.main import app

    img_path = _make_image(tmp_path / "in.jpg")
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(
        json.dumps({"detections": [{"box": [10, 10, 80, 80], "score": 0.7, "class_name": "obj"}]})
    )
    out_path = tmp_path / "out.jpg"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "draw",
            "image",
            "--image",
            str(img_path),
            "--pred",
            str(pred_path),
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()


def test_cli_annotate_image(tmp_path: Path) -> None:
    from visionservex.cli.main import app

    img_path = _make_image(tmp_path / "in.jpg")
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(
        json.dumps(
            {
                "task": "detect",
                "detections": [{"box": [10, 10, 80, 80], "score": 0.7, "class_name": "obj"}],
            }
        )
    )
    out_path = tmp_path / "out.jpg"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "annotate",
            "image",
            "--image",
            str(img_path),
            "--pred",
            str(pred_path),
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()


def test_cli_audit_syntax_debug(tmp_path: Path) -> None:
    from visionservex.cli.main import app

    out_json = tmp_path / "syntax.json"
    out_csv = tmp_path / "syntax.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "audit",
            "syntax-debug",
            "--limit",
            "10",
            "--out",
            str(out_json),
            "--csv-out",
            str(out_csv),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["n_models"] >= 1
    assert body["verdict"] in ("PASS", "FAIL")
    assert out_json.exists()
    assert out_csv.exists()


def test_manifest_has_live_fields() -> None:
    manifest_path = Path("docs/audit/visionservex_notebook_input_manifest.json")
    if not manifest_path.exists():
        pytest.skip("manifest not regenerated")
    manifest = json.loads(manifest_path.read_text())
    keys = {
        "draw_command",
        "live_supported",
        "video_supported",
        "expected_overlay_type",
        "recommended_live_source",
        "expected_fps_class",
    }
    for m in manifest["models"][:5]:
        assert keys.issubset(m.keys()), f"missing keys in {m.get('model_id')}: {keys - m.keys()}"


def test_benchmark_detection_requires_dataset() -> None:
    """benchmark-detection must refuse synthetic mode (no AP from fake data)."""
    from visionservex.cli.main import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark-detection",
            "--model",
            "dfine-s-o365-coco",
            "--dataset",
            "synthetic",
        ],
    )
    assert result.exit_code == 2
    assert "labelled dataset" in result.output.lower() or "labeled" in result.output.lower()


def test_version_is_2_14_0() -> None:
    import visionservex

    assert visionservex.__version__ == "2.14.0"
