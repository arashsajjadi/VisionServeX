# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.23.0 release-hardening: curated license rows, trap downgrades, coverage.

Locks in the strict pre-release decisions: license traps are out of the
commercial-safe set, flagship commercial-safe models are curated (not just
registry-derived), and docs examples use only curated commercial-safe models.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from visionservex import policy as P

pytestmark = pytest.mark.fast

_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# License traps downgraded OUT of commercial-safe
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mid",
    [
        "convnextv2-tiny",
        "convnextv2-base",
        "convnextv2-large",
        "rfdetr-seg-xlarge",
        "rfdetr-seg-2xlarge",
        "oneformer-swin-large",
    ],
)
def test_license_traps_are_not_commercial_safe(mid):
    pol = P.get_model_policy(mid)
    assert pol.is_commercial_safe is False, mid
    assert mid not in P.list_commercial_safe_models()
    assert pol.policy_source == "curated_override"
    assert pol.policy_notes  # documented reason


def test_convnextv2_conflict_is_documented():
    pol = P.get_model_policy("convnextv2-base")
    notes = pol.policy_notes.lower()
    assert "cc-by-nc" in notes and "conflict" in notes


# --------------------------------------------------------------------------- #
# Flagship commercial-safe models are CURATED (verified), not registry-derived
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mid",
    [
        "dfine-x-o365-coco",
        "rfdetr-base",
        "rfdetr-large",
        "grounding-dino-tiny",
        "florence-2-base",
        "clip-vit-large-patch14",
        "siglip2-base-patch16-224",
        "owlv2-base-patch16",
        "swinv2-base",
        "maxvit-tiny-tf-224",
        "sam2.1-hiera-small",
    ],
)
def test_flagship_commercial_safe_are_curated(mid):
    pol = P.get_model_policy(mid)
    assert pol.is_commercial_safe, mid
    assert pol.policy_source in ("curated_matrix", "curated_override"), (mid, pol.policy_source)


def test_dfine_and_rfdetr_have_verification_dates():
    for mid in ("dfine-x-o365-coco", "rfdetr-base"):
        assert P.get_model_policy(mid).last_verified_date == "2026-06-22"


# --------------------------------------------------------------------------- #
# Coverage: registry-derived commercial-safe are ONLY low-risk weight-less mocks
# --------------------------------------------------------------------------- #
def test_registry_derived_commercial_safe_are_only_mocks():
    rep = P.policy_coverage()
    leftover = [m for m in rep["registry_derived_commercial_safe_ids"] if not m.startswith("mock")]
    assert leftover == [], f"non-mock registry-derived commercial-safe models remain: {leftover}"
    assert rep["commercial_safe_curated_pct"] >= 90.0


def test_no_agpl_or_copyleft_in_commercial_safe():
    for mid in P.list_commercial_safe_models():
        pol = P.get_model_policy(mid)
        for lic in (pol.code_license, pol.weights_license):
            up = (lic or "").upper()
            assert "AGPL" not in up and "GPL-3" not in up and "GPLV3" not in up, (mid, lic)


def test_medsam2_still_excluded_and_research_only():
    assert P.get_model_policy("medsam2").commercial_status == "research_only"
    assert "medsam2" not in P.list_commercial_safe_models()
    assert "medsam2" in P.list_research_models()


# --------------------------------------------------------------------------- #
# Docs examples use only CURATED commercial-safe models (no registry-derived)
# --------------------------------------------------------------------------- #
def _model_ids_in(text: str) -> set[str]:
    ids: set[str] = set()
    ids |= set(re.findall(r'VisionModel\(\s*["\']([a-z0-9.\-]+)["\']', text))
    ids |= set(re.findall(r'assert_commercial_safe\(\s*["\']([a-z0-9.\-]+)["\']', text))
    ids |= set(
        re.findall(r"(?:predict|policy|explain|assert-commercial-safe)\s+([a-z][a-z0-9.\-]+)", text)
    )
    return ids


def test_docs_examples_use_only_curated_commercial_safe():
    known = set(P.list_models())
    for rel in ("docs/model_policy.md", "README.md", "docs/medical_segmentation.md"):
        p = _ROOT / rel
        if not p.exists():
            continue
        for mid in _model_ids_in(p.read_text(encoding="utf-8")):
            if mid not in known:
                continue
            pol = P.get_model_policy(mid)
            if pol.is_commercial_safe:
                # a commercial-safe model used in a doc example must be curated
                assert pol.policy_source != "registry_derived", f"{rel}: {mid} is registry-derived"


# --------------------------------------------------------------------------- #
# CLI surfaces
# --------------------------------------------------------------------------- #
def _app():
    from visionservex.cli.main import app

    return app


def test_cli_models_coverage():
    from typer.testing import CliRunner

    o = CliRunner().invoke(_app(), ["models", "coverage", "--json"])
    assert o.exit_code == 0
    rep = json.loads(o.stdout)
    assert rep["commercial_safe_curated_pct"] >= 90.0
    assert all(m.startswith("mock") for m in rep["registry_derived_commercial_safe_ids"])


def test_cli_models_byo_and_legal_review_filters():
    from typer.testing import CliRunner

    r = CliRunner()
    byo = {
        x["model_id"]
        for x in json.loads(r.invoke(_app(), ["models", "list", "--byo", "--json"]).stdout)
    }
    assert "sam3-base" in byo
    lr = {
        x["model_id"]
        for x in json.loads(r.invoke(_app(), ["models", "list", "--legal-review", "--json"]).stdout)
    }
    assert "convnextv2-base" in lr and "medsam" in lr
    # legal-review and commercial-safe are disjoint
    safe = set(P.list_commercial_safe_models())
    assert not (lr & safe)


def test_cli_predict_legal_review_refused_without_ack(tmp_path):
    from PIL import Image
    from typer.testing import CliRunner

    img = tmp_path / "x.png"
    Image.new("RGB", (8, 8)).save(img)
    o = CliRunner().invoke(_app(), ["predict", "convnextv2-base", str(img), "--json"])
    assert o.exit_code != 0
    assert "MODEL_NOT_COMMERCIAL_SAFE" in o.output
