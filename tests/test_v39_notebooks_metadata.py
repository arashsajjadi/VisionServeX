# SPDX-License-Identifier: Apache-2.0
"""v3.9 — v3.9 tutorial notebooks exist, have correct metadata, no raw token."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path("notebook/tutorials/v39_sam3_dinov3_byot_real_execution")
EXPECTED_NOTEBOOKS = [
    "01_install_from_pypi_and_hf_status.ipynb",
    "02_check_access_for_all_approved_gated_repos.ipynb",
    "03_dinov3_vits16_embedding.ipynb",
    "04_dinov3_vitb16_embedding.ipynb",
    "05_dinov3_convnext_tiny_embedding.ipynb",
    "06_dinov3_large_variants_status_and_resource_plan.ipynb",
    "07_sam3_image_text_prompt_segmentation.ipynb",
    "08_sam31_image_text_prompt_segmentation.ipynb",
    "09_sam31_video_status_or_smoke.ipynb",
    "10_byot_license_policy_and_nonredistribution.ipynb",
    "11_saco_safari_dataset_status_no_download.ipynb",
    "12_end_to_end_anastig_gated_model_policy_demo.ipynb",
]

_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{10,}")


def test_v39_notebooks_directory_exists():
    assert NOTEBOOKS_DIR.exists(), f"Missing directory: {NOTEBOOKS_DIR}"


@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_v39_notebook_exists(nb_name):
    nb_path = NOTEBOOKS_DIR / nb_name
    assert nb_path.exists(), f"Missing notebook: {nb_path}"


@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_v39_notebook_is_valid_json(nb_name):
    nb_path = NOTEBOOKS_DIR / nb_name
    if not nb_path.exists():
        pytest.skip(f"Notebook not yet created: {nb_name}")
    data = json.loads(nb_path.read_text())
    assert "cells" in data
    assert "metadata" in data


@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_v39_notebook_has_no_raw_token(nb_name):
    nb_path = NOTEBOOKS_DIR / nb_name
    if not nb_path.exists():
        pytest.skip(f"Notebook not yet created: {nb_name}")
    text = nb_path.read_text(errors="replace")
    tokens = _TOKEN_RE.findall(text)
    real = [t for t in tokens if not t.startswith("hf_***")]
    assert not real, f"Raw token in notebook {nb_name}: {real}"
