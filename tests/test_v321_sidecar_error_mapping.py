# SPDX-License-Identifier: Apache-2.0
"""v3.21: sidecar errors map to stable typed codes. Weight-free."""

from __future__ import annotations

import pytest

from visionservex.sidecars import SidecarClient
from visionservex.sidecars.errors import (
    SidecarError,
    SidecarNotConfigured,
    SidecarTimeout,
    SidecarUnavailable,
)


def test_error_codes_are_stable():
    assert SidecarNotConfigured("x").code == "SIDECAR_NOT_CONFIGURED"
    assert SidecarUnavailable("x").code == "SIDECAR_UNAVAILABLE"
    assert SidecarTimeout("x").code == "SIDECAR_TIMEOUT"
    assert SidecarError("x").code == "SIDECAR_ERROR"


def test_unconfigured_client_raises_not_configured():
    c = SidecarClient(None, name="florence2")
    from visionservex.sidecars.protocol import SidecarRequest

    with pytest.raises(SidecarNotConfigured):
        c.version()
    with pytest.raises(SidecarNotConfigured):
        c.predict(SidecarRequest(model_id="x", task="vlm", method="vlm"), image_bytes=b"")


def test_errors_are_subclasses_of_sidecar_error():
    for cls in (SidecarNotConfigured, SidecarUnavailable, SidecarTimeout):
        assert issubclass(cls, SidecarError)
