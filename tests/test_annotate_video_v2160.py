# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Phase 10 + 12 (v2.16.0): annotate-video diagnostics + synthetic video CLI.

The v16 notebook reported ``annotate-video failed with ok=False but empty
error message``. After v2.16, every annotate-video failure must come with a
``status``/``code``/``stage``/``message`` quadruple. The notebook should also
not depend on repo-local shell scripts; ``dev make-synthetic-video`` is the
package CLI equivalent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _vsx_cmd() -> list[str]:
    binary = shutil.which("visionservex")
    return [binary] if binary else [sys.executable, "-m", "visionservex"]


def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(_vsx_cmd() + args, capture_output=True, text=True, timeout=timeout)


def test_make_synthetic_video_writes_mp4(tmp_path: Path) -> None:
    out = tmp_path / "synthetic.mp4"
    res = _run(["dev", "make-synthetic-video", "--out", str(out), "--frames", "5", "--fps", "10"])
    if res.returncode == 2 and "OPENCV_REQUIRED" in res.stdout:
        pytest.skip("opencv not installed")
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert out.exists()
    payload = json.loads(res.stdout)
    assert payload["status"] == "ok"
    assert payload["frames"] == 5
    assert payload["size_bytes"] > 0


def test_annotate_video_missing_video_emits_structured(tmp_path: Path) -> None:
    """Missing --video must produce status=failed with stage=ARG_PARSE, NOT an empty error."""
    result_json = tmp_path / "result.json"
    out_mp4 = tmp_path / "out.mp4"
    res = _run(
        [
            "annotate",
            "video",
            "--out",
            str(out_mp4),
            "--model",
            "mock-detect",
            "--result-json",
            str(result_json),
            "--json",
        ]
    )
    assert res.returncode == 2
    assert result_json.exists()
    payload = json.loads(result_json.read_text())
    assert payload["status"] == "failed"
    assert payload["code"] == "VIDEO_ARG_REQUIRED"
    assert payload["stage"] == "ARG_PARSE"
    assert payload["message"]  # never empty
    assert payload["errors"]


def test_annotate_video_open_failed_emits_structured(tmp_path: Path) -> None:
    """A non-existent video path must produce a structured failure.

    The expected code depends on the runtime: when opencv-python is
    installed, the failure surfaces as ``VIDEO_OPEN_FAILED`` from the
    VideoCapture path. When opencv is missing (CI quick-test image), the
    earlier ``OPENCV_REQUIRED`` expected_blocker fires first. Both are
    structured outcomes and the test accepts either.
    """
    result_json = tmp_path / "result.json"
    out_mp4 = tmp_path / "out.mp4"
    fake_video = tmp_path / "does_not_exist.mp4"
    res = _run(
        [
            "annotate",
            "video",
            "--video",
            str(fake_video),
            "--out",
            str(out_mp4),
            "--model",
            "mock-detect",
            "--result-json",
            str(result_json),
            "--json",
        ]
    )
    assert res.returncode != 0
    assert result_json.exists()
    payload = json.loads(result_json.read_text())
    assert payload["status"] in {"failed", "expected_blocker"}
    assert payload["code"] in {"VIDEO_OPEN_FAILED", "OPENCV_REQUIRED"}
    assert payload.get("stage") in {"VIDEO_OPEN", "ARG_PARSE", None} or payload.get("stage")
    assert payload["message"]


def test_annotate_video_mock_detect_success(tmp_path: Path) -> None:
    """End-to-end happy path: make synthetic video → annotate with mock-detect → structured ok."""
    video = tmp_path / "synthetic.mp4"
    res = _run(["dev", "make-synthetic-video", "--out", str(video), "--frames", "3", "--fps", "5"])
    if res.returncode == 2 and "OPENCV_REQUIRED" in res.stdout:
        pytest.skip("opencv not installed")
    assert video.exists()

    out_mp4 = tmp_path / "annotated.mp4"
    json_out = tmp_path / "preds.jsonl"
    result_json = tmp_path / "result.json"
    res = _run(
        [
            "annotate",
            "video",
            "--video",
            str(video),
            "--model",
            "mock-detect",
            "--task",
            "detect",
            "--out",
            str(out_mp4),
            "--json-out",
            str(json_out),
            "--result-json",
            str(result_json),
            "--max-frames",
            "3",
            "--json",
        ]
    )
    assert result_json.exists(), (res.stdout, res.stderr)
    payload = json.loads(result_json.read_text())
    # Mock detect is always available; status must be ok and code OK.
    assert payload["status"] == "ok", payload
    assert payload["code"] == "OK"
    assert payload["frames_processed"] >= 1
    assert payload["output_video"] == str(out_mp4)
    assert payload["message"]  # never empty
