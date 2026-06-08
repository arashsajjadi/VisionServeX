# SPDX-License-Identifier: Apache-2.0
"""v3.8 — SAM-ViT-L/H are commercial-safe core and ONNX-exportable (or checkpoint-required)."""

from __future__ import annotations

from visionservex.licensing import policy as P
from visionservex.onnx_export import onnx_eligible


def test_sam_l_h_commercial_safe_core():
    for mid in ("sam-vit-large", "sam-vit-huge"):
        pol = P.get_policy(mid)
        assert pol.final_policy == "commercial_safe_core"
        assert pol.weights_license == "Apache-2.0"


def test_sam_l_h_aliases_resolve():
    assert P.resolve_model_id("sam-vit-l") == "sam-vit-large"
    assert P.resolve_model_id("sam-vit-h") == "sam-vit-huge"


def test_sam_l_h_onnx_eligible():
    elig = onnx_eligible()
    # v3.6 fix added L/H to ONNX eligibility
    assert {"sam-vit-l", "sam-vit-h"}.issubset(elig)
    assert "mobilesam" in elig and "sam-vit-b" in elig


def test_onnx_eligible_are_commercial_safe():
    for mid in onnx_eligible():
        pol = P.get_policy(mid)
        # mobilesam/sam-vit-* are all permissive core
        assert pol is not None
        assert pol.final_policy == "commercial_safe_core"
