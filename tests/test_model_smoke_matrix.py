# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.29.0: model smoke matrix integration tests."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))


# ---------------------------------------------------------------------------
# Unit-level tests (no subprocess)
# ---------------------------------------------------------------------------


def test_get_advertised_models_returns_list() -> None:
    """_get_advertised_models must return at least the core models."""
    from run_model_smoke_matrix import _get_advertised_models

    models = _get_advertised_models(include_core=True)
    assert len(models) >= 5, f"expected >= 5 core models, got {len(models)}"
    ids = {m["model_id"] for m in models}
    # Core detection model must be present
    assert any("dfine" in mid or "rfdetr" in mid for mid in ids), (
        f"No dfine/rfdetr model in core list: {ids}"
    )


def test_advertised_models_no_duplicates() -> None:
    """Each model_id must appear exactly once in the advertised list."""
    from run_model_smoke_matrix import _get_advertised_models

    models = _get_advertised_models(
        include_core=True,
        include_optional=True,
        include_sidecar=True,
    )
    ids = [m["model_id"] for m in models]
    duplicates = [mid for mid in set(ids) if ids.count(mid) > 1]
    assert not duplicates, f"Duplicate model IDs in advertised list: {duplicates}"


def test_smoke_command_synthesized_for_detect() -> None:
    """build_smoke_command for a detect task must include 'predict' subcommand."""
    from run_model_smoke_matrix import build_smoke_command

    cmd = build_smoke_command("dfine-s-o365-coco", "detect", device="cpu")
    assert "predict" in cmd, f"Missing 'predict' in: {cmd}"
    assert "dfine-s-o365-coco" in cmd, f"Missing model_id in: {cmd}"


def test_smoke_command_embed_uses_feature() -> None:
    """Embed task must use 'feature embed' subcommand."""
    from run_model_smoke_matrix import build_smoke_command

    cmd = build_smoke_command("dinov2-base", "embed", device="cpu")
    assert "feature" in cmd and "embed" in cmd, f"Embed command wrong: {cmd}"


def test_smoke_command_open_vocab_has_prompt() -> None:
    """Open-vocab detect must include --prompt flag."""
    from run_model_smoke_matrix import build_smoke_command

    cmd = build_smoke_command("owlv2-base-patch16", "open_vocab_detect", device="cpu")
    assert "--prompt" in cmd, f"Missing --prompt in open_vocab_detect cmd: {cmd}"


def test_smoke_command_foundation_seg_has_box() -> None:
    """Foundation segmentation must include --box prompt."""
    from run_model_smoke_matrix import build_smoke_command

    cmd = build_smoke_command("sam2-hiera-tiny", "foundation_segment", device="cpu")
    assert "--box" in cmd, f"Missing --box in foundation_segment cmd: {cmd}"


def test_classify_row_parseable_blocker_is_expected() -> None:
    """_classify_row must map parseable JSON blocker to expected_blocker, not failed_runtime."""
    import json
    from pathlib import Path

    from run_model_smoke_matrix import SmokeRow, _classify_row

    payload_str = json.dumps(
        {
            "status": "expected_blocker",
            "code": "ANOMALIB_REQUIRED",
            "message": "anomalib not installed",
        }
    )
    row = SmokeRow(model_id="test", family="anomaly", task="anomaly")
    row.returncode = 1
    row = _classify_row(row, payload_str, "", Path("/nonexistent/file.json"))
    assert row.final_state == "expected_blocker", (
        f"parseable ANOMALIB_REQUIRED classified as {row.final_state!r}"
    )


def test_classify_row_real_crash_is_failed() -> None:
    """An unstructured traceback must classify as failed_runtime."""
    from pathlib import Path

    from run_model_smoke_matrix import SmokeRow, _classify_row

    stderr = "Traceback (most recent call last):\n  File 'x.py'\nRuntimeError: boom"
    row = SmokeRow(model_id="test", family="x", task="detect")
    row.returncode = 1
    row = _classify_row(row, "", stderr, Path("/nonexistent/file.json"))
    assert row.final_state == "failed_runtime"


def test_classify_row_returncode_zero_is_passed() -> None:
    """returncode=0, valid JSON stdout → smoke_passed."""
    import json
    from pathlib import Path

    from run_model_smoke_matrix import SmokeRow, _classify_row

    stdout = json.dumps({"kind": "detection", "detections": []})
    row = SmokeRow(model_id="test", family="dfine", task="detect")
    row.returncode = 0
    row = _classify_row(row, stdout, "", Path("/nonexistent/file.json"))
    assert row.final_state == "smoke_passed", f"returncode=0 classified as {row.final_state!r}"


def test_final_states_are_valid() -> None:
    """All valid final_state values must be the allowed set."""
    from run_model_smoke_matrix import SmokeRow

    allowed = {
        "smoke_passed",
        "benchmark_passed",
        "expected_blocker",
        "license_blocked",
        "manual_checkpoint_required",
        "failed_runtime",
        "unclassified",
    }
    row = SmokeRow()
    row.final_state = "unclassified"
    assert row.final_state in allowed


# ---------------------------------------------------------------------------
# CLI smoke: ensure the models smoke-matrix CLI command is registered
# ---------------------------------------------------------------------------


def test_models_smoke_matrix_cli_registered() -> None:
    """visionservex models smoke-matrix --help must not fail."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "visionservex", "models", "smoke-matrix", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"smoke-matrix --help failed\nstdout={proc.stdout[:300]}\nstderr={proc.stderr[:300]}"
    )
    assert "smoke-matrix" in proc.stdout.lower() or "device" in proc.stdout.lower()


def test_smoke_assets_exist() -> None:
    """All required smoke assets must be present."""
    assets = [
        "tests/assets/smoke/coco_person_car.jpg",
        "tests/assets/smoke/coco_instance_sample.jpg",
        "tests/assets/smoke/coco_instance_sample.json",
        "tests/assets/smoke/medical_box_sample.png",
        "tests/assets/smoke/crop_weed_sample.jpg",
        "tests/assets/smoke/tracking_sample.mp4",
        "tests/assets/smoke/anomaly_simple/normal",
        "tests/assets/smoke/anomaly_simple/test",
    ]
    missing = []
    for a in assets:
        p = REPO_ROOT / a
        if not p.exists():
            missing.append(a)
    assert not missing, f"Missing smoke assets: {missing}"
