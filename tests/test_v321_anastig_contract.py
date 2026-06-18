# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.21: the committed Anastig contract is a faithful, complete projection of
``model_capabilities()`` including the new sidecar dimension. Weight-free.
"""

from __future__ import annotations

import json
from pathlib import Path

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import taxonomy as tx

ALLOWLIST = Path("docs/anastig_model_allowlist_v321.json")
CAPS = {m: model_capabilities(m) for m in list_models()}


def _contract() -> dict:
    return json.loads(ALLOWLIST.read_text())


def test_contract_exists_and_lists_buckets():
    assert ALLOWLIST.exists(), "run tools/qa/v321_build_artifacts.py"
    d = _contract()
    assert d["primary_partition_buckets"] and d["view_buckets"]


def test_primary_partition_is_disjoint_and_complete():
    d = _contract()
    primary = d["primary_partition_buckets"]
    seen: dict[str, str] = {}
    for b in primary:
        for mid in d[b]:
            assert mid not in seen, f"{mid} in two primary buckets: {seen.get(mid)} + {b}"
            seen[mid] = b
    assert set(seen) == set(CAPS), "primary partition must cover every model exactly once"


def test_sidecar_buckets_match_capabilities():
    d = _contract()
    sidecar_primary = (
        set(d["vlm_ready_live_sidecar"])
        | set(d["inference_ready_live_sidecar"])
        | set(d["segmentation_ready_live_sidecar"])
    )
    cap_sidecar = {
        m for m, c in CAPS.items() if c["readiness_state"] in tx.LIVE_SIDECAR_READY_STATES
    }
    assert sidecar_primary == cap_sidecar
    # the overlapping view equals the sidecar_required flag
    assert set(d["sidecar_required_live"]) == {m for m, c in CAPS.items() if c["sidecar_required"]}


def test_known_sidecar_models_are_present():
    d = _contract()
    assert set(d["vlm_ready_live_sidecar"]) == {"florence-2-base", "florence-2-large"}
    assert "rtmpose-m" in d["inference_ready_live_sidecar"]


def test_sam_decoder_fine_tune_view_matches_kind():
    d = _contract()
    assert set(d["sam_decoder_fine_tune_ready"]) == {
        m for m, c in CAPS.items() if c["fine_tune_kind"] == "frozen_encoder_decoder"
    }
    # all four HF SAM models, none of the SAM2 ones
    assert "sam-vit-base" in d["sam_decoder_fine_tune_ready"]
    assert not any(m.startswith("sam2") for m in d["sam_decoder_fine_tune_ready"])


def test_contract_version_matches_package():
    import visionservex as vsx

    assert _contract()["version"] == vsx.__version__
