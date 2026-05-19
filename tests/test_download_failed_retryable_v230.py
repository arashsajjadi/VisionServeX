# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.30.0: brotli/network download failures must be expected_blocker DOWNLOAD_FAILED."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "tools"))


def test_brotli_decoder_error_is_download_failed() -> None:
    from run_model_smoke_matrix import SmokeRow, _classify_row

    stderr = (
        "Traceback (most recent call last):\n"
        "  ...\n"
        "brotlicffi._api.error: brotli: decoder process called with data when "
        "'can_accept_more_data()' is False\n"
        "DownloadError: Hugging Face download failed for 'siglip-base-patch16-224'\n"
    )
    row = SmokeRow(model_id="siglip-base-patch16-224", family="siglip", task="embed")
    row.returncode = 1
    row = _classify_row(row, "", stderr, Path("/nonexistent.json"))
    assert row.final_state == "expected_blocker"
    assert row.blocker_code == "DOWNLOAD_FAILED"


def test_download_error_traceback_is_expected_blocker() -> None:
    from run_model_smoke_matrix import SmokeRow, _classify_row

    stderr = (
        "Traceback (most recent call last):\n"
        "  ...\n"
        "visionservex.runtime.downloads.DownloadError: Hugging Face download failed.\n"
    )
    row = SmokeRow(model_id="some-model", family="x", task="detect")
    row.returncode = 1
    row = _classify_row(row, "", stderr, Path("/nonexistent.json"))
    assert row.final_state == "expected_blocker"
    assert row.blocker_code == "DOWNLOAD_FAILED"


def test_predict_failed_brotli_message_is_expected_blocker() -> None:
    """Structured PREDICT_FAILED with brotli text → DOWNLOAD_FAILED."""
    from run_model_smoke_matrix import SmokeRow, _classify_row

    stderr = json.dumps(
        {
            "error": {
                "code": "PREDICT_FAILED",
                "message": "brotli: decoder process called with data when "
                "can_accept_more_data is False",
                "hint": "",
            }
        }
    )
    row = SmokeRow(model_id="swinv2-large", family="swinv2", task="classify")
    row.returncode = 1
    row = _classify_row(row, "", stderr, Path("/nonexistent.json"))
    assert row.final_state == "expected_blocker"
    assert row.blocker_code == "DOWNLOAD_FAILED"
