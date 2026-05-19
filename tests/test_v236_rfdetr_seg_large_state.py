# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.36.0: rfdetr-seg-large state must be documented."""

from __future__ import annotations


def test_rfdetr_seg_large_registry_entry() -> None:
    """rfdetr-seg-large must have an entry in the registry."""
    from visionservex.registry import default_registry

    reg = default_registry()
    e = reg.get("rfdetr-seg-large")
    assert e is not None, "rfdetr-seg-large missing from registry"
    assert e.task == "segment"
    assert e.family == "rfdetr"


def test_rfdetr_seg_large_has_license() -> None:
    from visionservex.registry import default_registry

    reg = default_registry()
    e = reg.get("rfdetr-seg-large")
    assert e is not None
    # rfdetr-seg-large uses PML 1.0 Plus license — must be noted
    assert e.license, "rfdetr-seg-large has no license recorded"
