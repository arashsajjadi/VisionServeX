# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Phase 1 (v2.17.0): `visionservex benchmark candidates --task detection`.

The notebook in v19 hand-built a candidate list that mixed mock-detect,
mock-open-vocab, and every D-FINE alias. v2.17.0 gives notebooks (and
CI) a single source of truth: the package decides what's eligible.

Tests pin: mocks excluded, aliases collapsed to canonical, sidecars +
unavailable + experimental excluded by default, open-vocab off by
default.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _vsx_cmd() -> list[str]:
    binary = shutil.which("visionservex")
    return [binary] if binary else [sys.executable, "-m", "visionservex"]


def _run_candidates(tmp_path: Path, *extra_args: str) -> dict:
    out = tmp_path / "candidates.json"
    res = subprocess.run(
        [
            *_vsx_cmd(),
            "benchmark",
            "candidates",
            "--task",
            "detection",
            "--scope",
            "clean",
            "--format",
            "json",
            "--out",
            str(out),
            *extra_args,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    return json.loads(out.read_text())


def test_clean_candidates_exclude_mock_detect(tmp_path: Path) -> None:
    payload = _run_candidates(tmp_path)
    row_ids = {r["model_id"] for r in payload["rows"]}
    assert "mock-detect" not in row_ids
    excluded_reasons = {(e["model_id"], e["reason"]) for e in payload["excluded"]}
    assert ("mock-detect", "MOCK_MODEL") in excluded_reasons


def test_clean_candidates_exclude_dfine_aliases(tmp_path: Path) -> None:
    """dfine-s and dfine-s-coco must not appear next to dfine-s-o365-coco."""
    payload = _run_candidates(tmp_path)
    row_ids = {r["model_id"] for r in payload["rows"]}
    assert "dfine-s" not in row_ids
    assert "dfine-s-coco" not in row_ids
    # Canonical id should be present
    assert "dfine-s-o365-coco" in row_ids
    # And the excluded entries must carry alias_of
    for e in payload["excluded"]:
        if e["model_id"] in ("dfine-s", "dfine-s-coco"):
            assert e["reason"] == "ALIAS_DUPLICATE"
            assert e["alias_of"] == "dfine-s-o365-coco"


def test_clean_candidates_have_canonical_metadata(tmp_path: Path) -> None:
    payload = _run_candidates(tmp_path)
    for row in payload["rows"]:
        assert row["model_id"] == row["canonical_model_id"]
        assert row["is_alias"] is False
        assert row["is_mock"] is False
        assert row["eligible_for_detection_benchmark"] is True
        assert row["expected_load_mode"] == "core_load"
        assert "family" in row
        assert row["model_size_key"] in {"n", "s", "m", "l", "x", "base", "huge", "unknown"}


def test_clean_candidates_exclude_unwired_stubs(tmp_path: Path) -> None:
    payload = _run_candidates(tmp_path)
    excluded_ids = {e["model_id"] for e in payload["excluded"]}
    # deim-*, deimv2-*, rtdetrv4-* are all stubs in the registry.
    assert "deim-s" in excluded_ids
    assert "rtdetrv4-s" in excluded_ids
    for e in payload["excluded"]:
        if e["model_id"].startswith(("deim", "rtdetrv4")):
            assert e["reason"] == "NOT_WIRED"


def test_clean_candidates_include_real_rfdetr(tmp_path: Path) -> None:
    payload = _run_candidates(tmp_path)
    row_ids = {r["model_id"] for r in payload["rows"]}
    # rfdetr-small / rfdetr-large / rfdetr-medium / rfdetr-nano / rfdetr-base
    # are real wired models — all must be in the clean list.
    expected = {"rfdetr-small", "rfdetr-large", "rfdetr-medium", "rfdetr-nano", "rfdetr-base"}
    assert expected.issubset(row_ids), row_ids


def test_clean_candidates_total_counts(tmp_path: Path) -> None:
    payload = _run_candidates(tmp_path)
    assert payload["n_rows"] == len(payload["rows"])
    assert payload["n_excluded"] == len(payload["excluded"])
    assert payload["task"] == "detection"
    assert payload["scope"] == "clean"
    assert payload["include_aliases"] is False


def test_open_vocab_excluded_by_default(tmp_path: Path) -> None:
    """Open-vocab task should not enter the closed-set detection list."""
    payload = _run_candidates(tmp_path)
    for row in payload["rows"]:
        assert "open" not in (row.get("family") or "").lower()


def test_include_aliases_keeps_alias_rows(tmp_path: Path) -> None:
    payload = _run_candidates(tmp_path, "--include-aliases")
    row_ids = {r["model_id"] for r in payload["rows"]}
    # With aliases included, both alias and canonical appear.
    assert "dfine-s" in row_ids
    assert "dfine-s-o365-coco" in row_ids


def test_unsupported_task_returns_expected_blocker(tmp_path: Path) -> None:
    out = tmp_path / "candidates.json"
    subprocess.run(
        [
            *_vsx_cmd(),
            "benchmark",
            "candidates",
            "--task",
            "open-vocab",
            "--format",
            "json",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(out.read_text())
    assert payload["status"] == "expected_blocker"
    assert payload["code"] == "TASK_NOT_SUPPORTED"
