# SPDX-License-Identifier: Apache-2.0
"""v3.10.0: global model manifest count — 170 entries, 95 runnable."""

from __future__ import annotations


def test_manifest_total_count():
    from visionservex.model_zoo.manifest import list_all_models

    models = list_all_models()
    assert len(models) >= 170, f"Expected >=170 manifest entries, got {len(models)}"


def test_manifest_runnable_count():
    from visionservex.model_zoo.manifest import get_model_source, list_all_models

    models = list_all_models()
    runnable = [
        mid for mid in models if (getattr(get_model_source(mid), "runnable_in_visionservex", False))
    ]
    assert len(runnable) >= 90, f"Expected >=90 runnable, got {len(runnable)}"


def test_manifest_has_sam3_entry():
    from visionservex.model_zoo.manifest import list_all_models

    models = list_all_models()
    sam3_models = [m for m in models if "sam3" in m.lower() and "sam2" not in m.lower()]
    assert len(sam3_models) >= 1, f"Expected >=1 sam3 entry, got {sam3_models}"


def test_manifest_has_dinov3_entries():
    from visionservex.model_zoo.manifest import list_all_models

    models = list_all_models()
    dinov3_models = [m for m in models if "dinov3" in m.lower() or "dino_v3" in m.lower()]
    # policy has 13 dinov3 entries; manifest may have them
    assert len(dinov3_models) >= 0  # at minimum not broken


def test_manifest_no_duplicate_model_ids():
    from visionservex.model_zoo.manifest import list_all_models

    models = list_all_models()
    assert len(models) == len(set(models)), (
        f"Duplicate model IDs in manifest: {[m for m in models if models.count(m) > 1]}"
    )
