# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Data utilities: COCO mappings, dataset loaders, dataset validators."""

from visionservex.data.coco_mapping import (
    COCO80_CONTIGUOUS_LABELS,
    COCO_CONTIGUOUS_TO_OFFICIAL,
    COCO_OFFICIAL_TO_CONTIGUOUS,
    is_official_id_set,
    remap_official_to_contiguous,
)

__all__ = [
    "COCO80_CONTIGUOUS_LABELS",
    "COCO_CONTIGUOUS_TO_OFFICIAL",
    "COCO_OFFICIAL_TO_CONTIGUOUS",
    "is_official_id_set",
    "remap_official_to_contiguous",
]
