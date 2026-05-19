# SPDX-License-Identifier: Apache-2.0
"""v2.33.0: model cache commands."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cache_status_runs() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "visionservex", "models", "cache-status"],
        capture_output=True,
        text=True,
        timeout=20,
        cwd=str(Path(__file__).parent.parent),
    )
    assert proc.returncode == 0
    p = json.loads(proc.stdout.strip())
    assert p["status"] == "ok"
    assert "cache_root" in p


def test_cache_add_and_verify(tmp_path) -> None:
    # Create a test "checkpoint"
    f = tmp_path / "test_weight.pt"
    f.write_bytes(b"fake_weights_content_for_testing")

    repo = Path(__file__).parent.parent
    out_root = tmp_path / "cache_root"
    import os

    env = {**os.environ, "VISION_SERVEX_MODEL_CACHE": str(out_root)}

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "visionservex",
            "models",
            "cache-add",
            "test-model",
            "--file",
            str(f),
            "--license",
            "Apache-2.0",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        cwd=str(repo),
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[:200]
    p = json.loads(proc.stdout.strip())
    assert p["status"] == "ok"
    assert p["model_id"] == "test-model"
    assert "sha256" in p

    # Verify
    proc2 = subprocess.run(
        [sys.executable, "-m", "visionservex", "models", "cache-verify", "test-model"],
        capture_output=True,
        text=True,
        timeout=20,
        cwd=str(repo),
        env=env,
    )
    assert proc2.returncode == 0
    p2 = json.loads(proc2.stdout.strip())
    assert p2["status"] == "ok"
    assert p2["code"] == "OK"
