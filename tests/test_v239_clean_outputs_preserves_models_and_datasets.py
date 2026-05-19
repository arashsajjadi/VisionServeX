# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.39.0: ``visionservex notebook clean-outputs`` must never delete
.venv, models/checkpoints, datasets, or the global model cache. Generated
reports/plots/visuals/commands must be deleted."""

from __future__ import annotations

import subprocess
import sys


def test_clean_outputs_dry_run_preserves_venv_and_checkpoints(tmp_path):
    fake = tmp_path / "notebook"
    (fake / ".venv" / "lib").mkdir(parents=True)
    (fake / ".venv" / "lib" / "important.bin").write_text("must-stay")
    (fake / "models" / "checkpoints").mkdir(parents=True)
    (fake / "models" / "checkpoints" / "weights.pt").write_text("MODEL")
    (fake / "datasets" / "coco").mkdir(parents=True)
    (fake / "datasets" / "coco" / "img.jpg").write_text("IMG")
    (fake / "01_task" / "reports").mkdir(parents=True)
    (fake / "01_task" / "reports" / "leaderboard.csv").write_text("model_id\n")
    (fake / "01_task" / "plots").mkdir()
    (fake / "01_task" / "plots" / "p.png").write_text("PNG")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "visionservex.__main__",
            "notebook",
            "clean-outputs",
            "--root",
            str(fake),
            "--dry-run",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "preserved_paths" in out
    # dry-run: nothing should actually be deleted
    assert (fake / ".venv" / "lib" / "important.bin").exists()
    assert (fake / "models" / "checkpoints" / "weights.pt").exists()
    assert (fake / "datasets" / "coco" / "img.jpg").exists()
    assert (fake / "01_task" / "reports" / "leaderboard.csv").exists()


def test_clean_outputs_real_run_deletes_reports_keeps_models(tmp_path):
    fake = tmp_path / "notebook"
    (fake / ".venv").mkdir(parents=True)
    (fake / "models" / "checkpoints").mkdir(parents=True)
    (fake / "models" / "checkpoints" / "weights.pt").write_text("MODEL")
    (fake / "datasets").mkdir()
    (fake / "datasets" / "data.json").write_text("DATA")
    (fake / "task" / "reports").mkdir(parents=True)
    (fake / "task" / "reports" / "leaderboard.csv").write_text("m\n")
    (fake / "task" / "plots").mkdir()
    (fake / "task" / "plots" / "p.png").write_text("PNG")
    (fake / "executed_run_EXECUTED.ipynb").write_text("{}")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "visionservex.__main__",
            "notebook",
            "clean-outputs",
            "--root",
            str(fake),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    # Preserved
    assert (fake / "models" / "checkpoints" / "weights.pt").exists()
    assert (fake / "datasets" / "data.json").exists()
    assert (fake / ".venv").exists()
    # Deleted
    assert not (fake / "task" / "reports" / "leaderboard.csv").exists()
    assert not (fake / "task" / "plots" / "p.png").exists()
    assert not (fake / "executed_run_EXECUTED.ipynb").exists()
