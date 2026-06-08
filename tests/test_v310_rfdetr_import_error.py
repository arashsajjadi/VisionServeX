# SPDX-License-Identifier: Apache-2.0
"""v3.10.0: rfdetr_seg_runtime BUG_A2 fix — actionable ImportError."""
from __future__ import annotations

import sys

import pytest


def test_rfdetr_segment_importerror_has_hint():
    """When rfdetr is absent, segment_instances raises ImportError with pip hint."""
    rfdetr_installed = "rfdetr" in sys.modules or _rfdetr_importable()
    if rfdetr_installed:
        pytest.skip("rfdetr is installed; cannot test missing-package path")

    import numpy as np
    from PIL import Image

    from visionservex.rfdetr_seg_runtime import segment_instances

    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    with pytest.raises(ImportError) as exc_info:
        segment_instances("rfdetr-seg-small", img)
    msg = str(exc_info.value)
    assert "pip install" in msg, f"ImportError lacks install hint: {msg}"
    assert "visionservex[rfdetr]" in msg, f"ImportError missing extra name: {msg}"


def _rfdetr_importable() -> bool:
    try:
        import importlib

        importlib.import_module("rfdetr")
        return True
    except ImportError:
        return False


def test_rfdetr_unknown_variant_raises_valueerror():
    import numpy as np
    from PIL import Image

    from visionservex.rfdetr_seg_runtime import segment_instances

    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="unknown RF-DETR-Seg variant"):
        segment_instances("rfdetr-seg-nonexistent", img)


def test_rfdetr_explain_known_variant():
    from visionservex.rfdetr_seg_runtime import explain

    result = explain("rfdetr-seg-small")
    assert result["state"] == "benchmark_passed"
    assert result["license"] == "Apache-2.0"
    assert result["commercial_safe"] is True


def test_rfdetr_variants_list():
    from visionservex.rfdetr_seg_runtime import variants

    v = variants()
    assert "rfdetr-seg-small" in v
    assert "rfdetr-seg-nano" in v
    assert "rfdetr-seg-2xl" in v
    assert len(v) == 6
