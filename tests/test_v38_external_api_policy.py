# SPDX-License-Identifier: Apache-2.0
"""v3.8 — external-API models are never counted as local models."""

from __future__ import annotations

import pytest

from visionservex.licensing import policy as P

API = [
    r.model_id for r in P.iter_policies() if r.final_policy == "external_api_only_terms_required"
]


def test_api_models_present():
    assert "grounding-dino-1.5-pro" in API
    assert "dino-x-api" in API


@pytest.mark.parametrize("mid", API)
def test_api_invariants(mid):
    pol = P.get_policy(mid)
    assert pol.is_local is False
    assert pol.production_allowed is False
    assert pol.default_safe is False
    assert pol.can_ship_weights is False
    assert "API" in pol.warning_text or "provider" in pol.warning_text.lower()


def test_api_not_counted_as_local():
    local = [r for r in P.iter_policies() if r.is_local]
    assert all(r.final_policy != "external_api_only_terms_required" for r in local)


def test_local_count_excludes_api():
    n_api = len(API)
    n_local = sum(1 for r in P.iter_policies() if r.is_local)
    n_total = sum(1 for _ in P.iter_policies())
    assert n_local == n_total - n_api
