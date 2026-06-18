# SPDX-License-Identifier: Apache-2.0
"""v3.21: generic sidecar request/response protocol. Weight-free."""

from __future__ import annotations

import pytest

from visionservex.sidecars import SidecarClient, SidecarRequest, SidecarResponse
from visionservex.sidecars.errors import SidecarRequestInvalid, SidecarResponseInvalid


def test_valid_request_round_trips():
    req = SidecarRequest(model_id="florence-2-base", task="vlm", method="vlm", text="caption")
    assert req.validate() is req
    fields = req.form_fields()
    assert fields["model_id"] == "florence-2-base" and fields["method"] == "vlm"


def test_invalid_request_raises_typed():
    with pytest.raises(SidecarRequestInvalid):
        SidecarRequest(model_id="", task="vlm", method="vlm").validate()
    with pytest.raises(SidecarRequestInvalid):
        SidecarRequest(model_id="x", task="vlm", method="bogus").validate()
    with pytest.raises(SidecarRequestInvalid):
        SidecarRequest(model_id="x", task="", method="predict").validate()


def test_response_normalises_and_requires_model_id():
    r = SidecarResponse.from_json({"model_id": "florence-2-base", "task": "vlm", "text": "a cat"})
    assert r.model_id == "florence-2-base"
    assert r.payload["text"] == "a cat"
    with pytest.raises(SidecarResponseInvalid):
        SidecarResponse.from_json({"task": "vlm"})  # missing model_id
    with pytest.raises(SidecarResponseInvalid):
        SidecarResponse.from_json(["not", "a", "dict"])


def test_client_disabled_without_url():
    c = SidecarClient(None, name="florence2")
    assert not c.configured
    assert c.health() is False
