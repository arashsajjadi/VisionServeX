# SPDX-License-Identifier: Apache-2.0
"""v3.8 — SAM2.1 ONNX export attempt is honest (structured, never faked)."""

from __future__ import annotations

import os

import pytest

from visionservex import VSX, VSXError
from visionservex.licensing import policy as P
from visionservex.onnx_export import onnx_eligible


def test_sam21_is_commercial_safe_core():
    pol = P.get_policy("sam2.1-hiera-small")
    assert pol.final_policy == "commercial_safe_core"


def test_sam21_onnx_not_eligible_is_honest():
    """SAM2.1 is not in the ONNX decoder-export set today — the attempt must
    return a structured 'not_applicable', not a fabricated success."""
    assert "sam2.1-hiera-small" not in onnx_eligible()
    with pytest.raises(VSXError) as exc:
        VSX.sam("sam2.1-hiera-small").to_onnx("/tmp/should_not_be_written.onnx")
    assert exc.value.state == "not_applicable"


@pytest.mark.sam_onnx
@pytest.mark.skipif(
    os.environ.get("VISIONSERVEX_RUN_DOWNLOAD_TESTS") != "1",
    reason="set VISIONSERVEX_RUN_DOWNLOAD_TESTS=1 to attempt a real ONNX export",
)
def test_sam_vit_b_onnx_export_real(tmp_path):
    out = tmp_path / "mobilesam.onnx"
    VSX.sam("mobilesam").to_onnx(str(out))
    assert out.exists() and out.stat().st_size > 0
