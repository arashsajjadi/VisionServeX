# SPDX-License-Identifier: Apache-2.0
"""v3.8 — commercial-safe core invariants + the cross-cutting hard rules."""

from __future__ import annotations

import pytest

from visionservex.licensing import policy as P

CORE = [r.model_id for r in P.iter_policies() if r.final_policy == "commercial_safe_core"]


def test_core_has_expected_members():
    for m in ("sam-vit-base", "sam2.1-hiera-large", "dinov2-base",
              "florence-2-base", "clip-vit-base-patch32", "rfdetr-seg-small"):
        assert m in CORE


@pytest.mark.parametrize("mid", CORE)
def test_core_invariants(mid):
    pol = P.get_policy(mid)
    assert pol.default_safe is True
    assert pol.commercial_safe is True
    assert pol.production_allowed is True
    assert pol.gated is False
    assert pol.can_ship_weights is False  # never bundle weights
    assert pol.can_auto_download is True


def test_no_row_can_ship_weights():
    assert all(not r.can_ship_weights for r in P.iter_policies())


def test_noncommercial_never_production():
    for r in P.iter_policies():
        if r.final_policy in ("noncommercial_restricted",):
            assert not r.production_allowed and not r.default_safe


def test_legal_review_never_commercial_safe():
    for r in P.iter_policies():
        if r.final_policy == "legal_review_required":
            assert not r.commercial_safe


def test_every_warning_is_canonical():
    valid = set(P.WARNING_TEXTS.values())
    for r in P.iter_policies():
        assert r.warning_text in valid


def test_final_policy_vocabulary_closed():
    for r in P.iter_policies():
        assert r.final_policy in P.FINAL_POLICIES


def test_matrix_columns_stable():
    rows = P.matrix_rows()
    from visionservex.licensing.policy import MATRIX_COLUMNS

    assert rows and set(rows[0].keys()) == set(MATRIX_COLUMNS)
