# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.25.0: Domain dataset registry + validators.

VisionServeX is honest about which models can be benchmarked on which
datasets. A model that says ``task=segment`` in the manifest cannot be
benchmarked on COCO128 detection labels and then claim segmentation AP.
The :mod:`visionservex.datasets.domain_registry` lists every specialized
domain we integrate and the dataset shape/metric it actually needs.
"""

from visionservex.datasets.domain_registry import (
    DOMAIN_REGISTRY,
    DomainDataset,
    list_domain_datasets,
    validate_domain_path,
)

__all__ = [
    "DOMAIN_REGISTRY",
    "DomainDataset",
    "list_domain_datasets",
    "validate_domain_path",
]
