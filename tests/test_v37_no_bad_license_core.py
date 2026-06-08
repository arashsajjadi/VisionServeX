# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7: no AGPL/GPL/NC/proprietary/gated model may be commercial-safe in core."""

from __future__ import annotations

import csv
from pathlib import Path

R = Path(__file__).parent.parent / "notebook" / "99_final_report" / "reports"

BAD_TOKENS = [
    "AGPL",
    "GPL",
    "non-commercial",
    "NON-COMMERCIAL",
    "NC",
    "proprietary",
    "research-only",
    "S-Lab",
    "NVIDIA License",
    "DINOv3 License",
    "SAM License",
]


def _matrix(name, key):
    return list(csv.DictReader((R / name).open()))


def test_sam_matrix_no_bad_license_commercial_safe():
    for r in _matrix("v37_sam_variant_matrix.csv", "variant_id"):
        if str(r["commercial_safe"]).lower() == "true":
            lic = r["license"]
            # Apache/MIT/BSD only in commercial-safe; "Apache-2.0 code / HQSeg-44K NC data"
            # rows must NOT be commercial_safe=True
            assert "NC" not in lic and "non-commercial" not in lic.lower(), (
                f"{r['variant_id']} commercial_safe but license={lic}"
            )


def test_dino_matrix_no_bad_license_commercial_safe():
    for r in _matrix("v37_dino_variant_matrix.csv", "variant_id"):
        if str(r["commercial_safe"]).lower() == "true":
            lic = r["license"]
            assert "DINOv3 License" not in lic and "proprietary" not in lic.lower(), (
                f"{r['variant_id']} commercial_safe but license={lic}"
            )
            assert lic == "Apache-2.0", (
                f"{r['variant_id']} commercial-safe must be Apache, got {lic}"
            )


def test_inventory_commercial_safe_only_permissive():
    for r in _matrix("v37_post_v259_inventory.csv", "item_id"):
        if str(r["commercial_safe"]).lower() == "true":
            lic = r["license_status"]
            for tok in [
                "AGPL",
                "non-commercial",
                "NVIDIA License",
                "DINOv3 License",
                "S-Lab",
                "proprietary",
            ]:
                assert tok not in lic, (
                    f"{r['item_id']} commercial_safe with bad license token {tok!r}: {lic}"
                )


def test_locateanything_edgesam_fastsam_yolo_not_safe():
    inv = {r["item_id"]: r for r in _matrix("v37_post_v259_inventory.csv", "item_id")}
    for m in ["locateanything-3b", "edgesam", "fastsam-s", "fastsam-x", "yolov8-seg", "yolo11-seg"]:
        if m in inv:
            assert str(inv[m]["commercial_safe"]).lower() == "false"


def test_dinov3_never_labeled_apache():
    for r in _matrix("v37_dino_variant_matrix.csv", "variant_id"):
        if r["variant_id"].startswith("dinov3"):
            assert "Apache" not in r["license"]
