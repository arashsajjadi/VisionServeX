# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.15.0 notebook CLI contract tests.

Verifies that every notebook-facing command accepts the exact option set
that the v16 Colab notebook generates. No model weights are loaded.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _cli(*args: str) -> tuple[int, str, str]:
    """Run visionservex CLI and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "visionservex", *args],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def _no_usage_error(stderr: str) -> bool:
    """Return True if there is no 'No such option' usage error."""
    return "No such option" not in stderr and "Got unexpected extra argument" not in stderr


# ============================================================
# Phase 1 — detect / open-vocab option contract
# ============================================================


def test_detect_accepts_out_and_draw(tmp_path):
    """visionservex detect MODEL IMAGE --out JSON --draw IMAGE must be accepted."""
    _rc, _out, err = _cli(
        "detect",
        "mock-detect",
        "examples/images/street.jpg",
        "--conf",
        "0.25",
        "--out",
        str(tmp_path / "det.json"),
        "--draw",
        str(tmp_path / "det.jpg"),
        "--device",
        "cpu",
    )
    assert _no_usage_error(err), f"CLI usage error: {err[:300]}"


def test_detect_still_accepts_save_json_and_save_image(tmp_path):
    """Legacy --save-json and --save-image must still be accepted."""
    _rc, _out, err = _cli(
        "detect",
        "mock-detect",
        "examples/images/street.jpg",
        "--save-json",
        str(tmp_path / "det.json"),
        "--save-image",
        str(tmp_path / "det.jpg"),
    )
    assert _no_usage_error(err), f"CLI usage error: {err[:300]}"


def test_open_vocab_accepts_out_and_draw(tmp_path):
    """visionservex open-vocab MODEL IMAGE --prompt ... --out JSON --draw IMAGE."""
    _rc, _out, err = _cli(
        "open-vocab",
        "mock-open-vocab",
        "examples/images/street.jpg",
        "--prompt",
        "person, car",
        "--out",
        str(tmp_path / "ov.json"),
        "--draw",
        str(tmp_path / "ov.jpg"),
    )
    assert _no_usage_error(err), f"CLI usage error: {err[:300]}"


# ============================================================
# Phase 2 — audit syntax-debug option contract
# ============================================================


def test_audit_syntax_debug_accepts_manifest_option(tmp_path):
    """--manifest must be accepted."""
    manifest = Path("docs/audit/visionservex_notebook_input_manifest.json")
    if not manifest.exists():
        pytest.skip("manifest not present")
    rc, _out, err = _cli(
        "audit",
        "syntax-debug",
        "--manifest",
        str(manifest),
        "--out",
        str(tmp_path / "debug.json"),
    )
    assert _no_usage_error(err), f"No such option error: {err[:300]}"
    assert rc == 0


def test_audit_syntax_debug_accepts_all_flag(tmp_path):
    """--all must be accepted."""
    manifest = Path("docs/audit/visionservex_notebook_input_manifest.json")
    if not manifest.exists():
        pytest.skip("manifest not present")
    rc, _out, err = _cli(
        "audit",
        "syntax-debug",
        "--manifest",
        str(manifest),
        "--all",
        "--out",
        str(tmp_path / "debug.json"),
    )
    assert _no_usage_error(err), f"No such option error: {err[:300]}"
    assert rc == 0


def test_audit_syntax_debug_accepts_resource_guard(tmp_path):
    """--resource-guard must be accepted."""
    manifest = Path("docs/audit/visionservex_notebook_input_manifest.json")
    if not manifest.exists():
        pytest.skip("manifest not present")
    rc, _out, err = _cli(
        "audit",
        "syntax-debug",
        "--manifest",
        str(manifest),
        "--resource-guard",
        "--out",
        str(tmp_path / "debug.json"),
    )
    assert _no_usage_error(err), f"No such option error: {err[:300]}"
    assert rc == 0


def test_audit_syntax_debug_accepts_draw_dir(tmp_path):
    """--draw-dir must be accepted."""
    manifest = Path("docs/audit/visionservex_notebook_input_manifest.json")
    if not manifest.exists():
        pytest.skip("manifest not present")
    rc, _out, err = _cli(
        "audit",
        "syntax-debug",
        "--manifest",
        str(manifest),
        "--draw-dir",
        str(tmp_path / "draws"),
        "--out",
        str(tmp_path / "debug.json"),
    )
    assert _no_usage_error(err), f"No such option error: {err[:300]}"
    assert rc == 0


def test_audit_syntax_debug_accepts_max_models_per_family(tmp_path):
    """--max-models-per-family must be accepted (quick mode)."""
    manifest = Path("docs/audit/visionservex_notebook_input_manifest.json")
    if not manifest.exists():
        pytest.skip("manifest not present")
    rc, _out, err = _cli(
        "audit",
        "syntax-debug",
        "--manifest",
        str(manifest),
        "--max-models-per-family",
        "3",
        "--out",
        str(tmp_path / "debug.json"),
    )
    assert _no_usage_error(err), f"No such option error: {err[:300]}"
    assert rc == 0


def test_audit_syntax_debug_writes_json(tmp_path):
    """Output must be valid JSON with new schema fields."""
    manifest = Path("docs/audit/visionservex_notebook_input_manifest.json")
    if not manifest.exists():
        pytest.skip("manifest not present")
    out_path = tmp_path / "debug.json"
    rc, _out, _err = _cli(
        "audit",
        "syntax-debug",
        "--manifest",
        str(manifest),
        "--out",
        str(out_path),
    )
    assert rc == 0
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "status" in data
    assert "summary" in data
    assert "rows" in data


# ============================================================
# Phase 3 — annotate image / video option contract
# ============================================================


def test_annotate_image_accepts_model_and_task(tmp_path):
    """annotate image --model --task --json-out must be accepted."""
    _rc, _out, err = _cli(
        "annotate",
        "image",
        "--model",
        "mock-detect",
        "--image",
        "examples/images/street.jpg",
        "--task",
        "detect",
        "--out",
        str(tmp_path / "annotated.jpg"),
        "--json-out",
        str(tmp_path / "annotated.json"),
    )
    assert _no_usage_error(err), f"No such option error: {err[:300]}"


def test_annotate_image_pred_mode_still_works():
    """Legacy --pred mode must still be accepted (--pred is now optional).

    v2.25.1: use rich-aware help matcher (CI terminal soft-wraps long lines).
    """
    from tests.helpers.cli_help import assert_help_contains_all, run_help

    res = run_help(["annotate", "image"])
    assert_help_contains_all(res, ["--pred", "--model"])


def test_annotate_video_accepts_model_and_task():
    """annotate video --model --task --json-out --max-frames must be accepted."""
    from tests.helpers.cli_help import assert_help_contains_all, run_help

    res = run_help(["annotate", "video"])
    assert_help_contains_all(res, ["--model", "--task", "--json-out", "--max-frames"])
    assert _no_usage_error(res.stderr)


def test_annotate_video_no_usage_error(tmp_path):
    """annotate video with all notebook options must not emit 'No such option'."""
    _rc, _out, err = _cli(
        "annotate",
        "video",
        "--model",
        "mock-detect",
        "--video",
        "/tmp/nonexistent_video.mp4",
        "--task",
        "detect",
        "--out",
        str(tmp_path / "out.mp4"),
        "--json-out",
        str(tmp_path / "out.jsonl"),
        "--max-frames",
        "10",
    )
    assert _no_usage_error(err), f"Usage error: {err[:300]}"


# ============================================================
# Phase 8 — validate/doctor --format json --out contract
# ============================================================


def test_medical_validate_accepts_format_and_out(tmp_path):
    """medical validate MODEL --format json --out PATH must be accepted."""
    out_path = tmp_path / "result.json"
    _rc, _out, err = _cli(
        "medical",
        "validate",
        "totalsegmentator",
        "--format",
        "json",
        "--out",
        str(out_path),
    )
    assert _no_usage_error(err), f"No such option: {err[:300]}"
    assert out_path.exists(), "Output file must be written"
    data = json.loads(out_path.read_text())
    assert "model_id" in data
    assert "status" in data


def test_medical_validate_nnunet(tmp_path):
    out_path = tmp_path / "result.json"
    _rc, _out, err = _cli(
        "medical", "validate", "nnunet-v2", "--format", "json", "--out", str(out_path)
    )
    assert _no_usage_error(err)
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "status" in data


def test_medical_monai_list_bundles_accepts_format_and_out(tmp_path):
    out_path = tmp_path / "monai.json"
    _rc, _out, err = _cli(
        "medical", "monai", "list-bundles", "--format", "json", "--out", str(out_path)
    )
    assert _no_usage_error(err), f"No such option: {err[:300]}"
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "status" in data
    assert "code" in data


def test_agriculture_doctor_accepts_format_and_out(tmp_path):
    out_path = tmp_path / "agri.json"
    rc, _out, err = _cli("agriculture", "doctor", "--format", "json", "--out", str(out_path))
    assert _no_usage_error(err), f"No such option: {err[:300]}"
    assert rc == 0
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "components" in data


def test_agriculture_prompt_detect_accepts_draw():
    """agriculture prompt-detect must accept --draw option."""
    from tests.helpers.cli_help import assert_help_contains_all, run_help

    res = run_help(["agriculture", "prompt-detect"])
    assert_help_contains_all(res, ["--draw"])


def test_agriculture_prompt_segment_accepts_draw():
    """agriculture prompt-segment must accept --draw option."""
    from tests.helpers.cli_help import assert_help_contains_all, run_help

    res = run_help(["agriculture", "prompt-segment"])
    assert_help_contains_all(res, ["--draw"])


def test_openmmlab_validate_accepts_format_and_out(tmp_path):
    out_path = tmp_path / "result.json"
    _rc, _out, err = _cli(
        "openmmlab", "validate", "rtmpose-m", "--format", "json", "--out", str(out_path)
    )
    assert _no_usage_error(err), f"No such option: {err[:300]}"
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "status" in data
    assert "model_id" in data


def test_maskdino_validate_accepts_format_and_out(tmp_path):
    out_path = tmp_path / "result.json"
    _rc, _out, err = _cli(
        "maskdino", "validate", "maskdino-swinl-coco", "--format", "json", "--out", str(out_path)
    )
    assert _no_usage_error(err), f"No such option: {err[:300]}"
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "status" in data
    assert "model_id" in data


def test_sam_family_validate_accepts_format_and_out(tmp_path):
    out_path = tmp_path / "result.json"
    _rc, _out, err = _cli(
        "sam-family", "validate", "sam3.1", "--format", "json", "--out", str(out_path)
    )
    assert _no_usage_error(err), f"No such option: {err[:300]}"
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "status" in data


# ============================================================
# No raw traceback tests
# ============================================================


def test_detect_unknown_model_no_raw_traceback():
    """detect with unknown model must not produce raw Python traceback."""
    _rc, out, err = _cli("detect", "nonexistent-model-xyz", "examples/images/street.jpg")
    combined = out + err
    assert "Traceback (most recent call last)" not in combined


def test_medical_validate_unknown_model_returns_json(tmp_path):
    """medical validate with unknown model must return JSON with error code."""
    out_path = tmp_path / "result.json"
    _rc, _out, _err = _cli(
        "medical",
        "validate",
        "nonexistent-medical-model",
        "--format",
        "json",
        "--out",
        str(out_path),
    )
    assert out_path.exists(), "Output file must be written even for unknown model"
    data = json.loads(out_path.read_text())
    assert "code" in data
    assert data["code"] == "UNKNOWN_MEDICAL_MODEL"


# ============================================================
# Canonical JSON output schema tests
# ============================================================


def test_detection_output_has_detections_key(tmp_path):
    """detect with mock model must write JSON with 'detections' key."""
    out_path = tmp_path / "out.json"
    rc, _out, _err = _cli(
        "detect",
        "mock-detect",
        "examples/images/street.jpg",
        "--out",
        str(out_path),
    )
    assert rc == 0
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "detections" in data


def test_validate_json_has_canonical_fields(tmp_path):
    """validate commands must produce canonical status/code/message fields."""
    out_path = tmp_path / "result.json"
    _rc, _out, _err = _cli(
        "medical", "validate", "totalsegmentator", "--format", "json", "--out", str(out_path)
    )
    data = json.loads(out_path.read_text())
    for field in ("model_id", "status", "code", "message"):
        assert field in data, f"Missing field '{field}' in validate JSON output"


# ============================================================
# medical segment --draw contract
# ============================================================


def test_medical_segment_accepts_draw_option():
    """medical segment must accept --draw option."""
    from tests.helpers.cli_help import assert_help_contains_all, run_help

    res = run_help(["medical", "segment"])
    assert_help_contains_all(res, ["--draw"])
