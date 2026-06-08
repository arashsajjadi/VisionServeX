# SPDX-License-Identifier: Apache-2.0
"""v3.11: INSID3 registry matrix — 3 variants, correct backbones, no weight shipping."""

from __future__ import annotations


def _insid3_rows():
    from visionservex.licensing.policy import POLICY

    return {k: v for k, v in POLICY.items() if v.family == "insid3"}


def test_registry_has_three_variants():
    rows = _insid3_rows()
    assert len(rows) == 3


def test_registry_backbones_match_size():
    rows = _insid3_rows()
    small = rows.get("insid3-small")
    base = rows.get("insid3-base")
    large = rows.get("insid3-large")
    assert small is not None and "vits16" in small.hf_repo
    assert base is not None and "vitb16" in base.hf_repo
    assert large is not None and "vitl16" in large.hf_repo


def test_all_insid3_can_ship_weights_false():
    rows = _insid3_rows()
    for mid, pol in rows.items():
        assert not pol.can_ship_weights, f"{mid} can_ship_weights must be False"


def test_all_insid3_gated():
    rows = _insid3_rows()
    for mid, pol in rows.items():
        assert pol.gated, f"{mid} must be gated=True (BYOT)"


def test_all_insid3_local_token_required():
    rows = _insid3_rows()
    for mid, pol in rows.items():
        assert pol.local_token_required, f"{mid} must require local token"


def test_global_policy_row_count_increased():
    from visionservex.licensing.policy import _ROWS

    assert len(_ROWS) >= 102, (
        f"Expected >=102 policy rows after INSID3 addition (99 + 3), got {len(_ROWS)}"
    )
