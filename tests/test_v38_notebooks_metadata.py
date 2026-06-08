# SPDX-License-Identifier: Apache-2.0
"""v3.8 — tutorial notebooks exist, are valid, assert site-packages, and leak no token."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

NB_DIR = Path("notebook/tutorials/v38_hf_byot_and_license_safe_models")
EXPECTED = [
    "01_install_from_pypi_and_check_version.ipynb",
    "02_connect_huggingface_token_safely.ipynb",
    "03_license_policy_matrix_explained.ipynb",
    "04_pull_commercial_safe_sam_and_dino_models.ipynb",
    "05_sam3_byot_status_and_optional_run.ipynb",
    "06_dinov3_byot_status_and_optional_embedding.ipynb",
    "07_sam2_1_onnx_export_attempt.ipynb",
    "08_ritm_interactive_segmentation_checkpoint_path.ipynb",
    "09_rfdetr_seg_commercial_safe_instance_masks.ipynb",
    "10_groundingdino_sam_text_to_mask_pipeline.ipynb",
    "11_restricted_models_warnings_locateanything_edgesam_fastsam_yolo.ipynb",
    "12_end_to_end_anastig_policy_demo.ipynb",
]

TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")


@pytest.mark.skipif(not NB_DIR.exists(), reason="v38 notebooks dir not present")
def test_all_twelve_notebooks_present():
    have = {p.name for p in NB_DIR.glob("*.ipynb")}
    missing = [n for n in EXPECTED if n not in have]
    assert not missing, f"missing notebooks: {missing}"


@pytest.mark.skipif(not NB_DIR.exists(), reason="v38 notebooks dir not present")
@pytest.mark.parametrize("name", EXPECTED)
def test_notebook_is_valid_and_clean(name):
    path = NB_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    nb = json.loads(path.read_text())
    assert nb.get("cells"), f"{name} has no cells"
    src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    # never embeds a real token
    assert not TOKEN_RE.search(src), f"{name} contains a token-like string"


@pytest.mark.skipif(not NB_DIR.exists(), reason="v38 notebooks dir not present")
def test_notebooks_assert_site_packages():
    # every notebook should verify it imports the installed package, not local src
    for name in EXPECTED:
        path = NB_DIR / name
        if not path.exists():
            continue
        nb = json.loads(path.read_text())
        src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
        assert "site-packages" in src, f"{name} does not assert site-packages"
