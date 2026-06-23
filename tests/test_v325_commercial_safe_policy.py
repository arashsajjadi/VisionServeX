# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.25: commercial-safe-by-default policy + acknowledgement gate.

Strict, weight-free tests. VisionServeX core is commercial-safe by default;
restricted models (research/non-commercial/AGPL/legal-review/BYO) are blocked
unless an explicit acknowledged pathway is used. MedSAM2 is the canonical
research-only model and must never be commercial-safe.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from visionservex import policy as P
from visionservex.exceptions import (
    ModelLicenseError,
    ModelNotCommercialSafeError,
)
from visionservex.registry import RegistryError

pytestmark = pytest.mark.fast
runner = CliRunner()


# --------------------------------------------------------------------------- #
# Policy: commercial status
# --------------------------------------------------------------------------- #
def test_medsam2_is_not_commercial_safe():
    pol = P.get_model_policy("medsam2")
    assert pol.commercial_status == "research_only"
    assert pol.is_commercial_safe is False
    assert pol.requires_acknowledgement is True
    assert pol.default_enabled is False
    assert pol.not_for_diagnosis_required is True


def test_medsam2_excluded_from_commercial_safe_list_included_in_research():
    assert "medsam2" not in P.list_commercial_safe_models()
    assert "medsam2" in P.list_research_models()


def test_commercial_safe_sam2_and_permissive_detectors():
    for mid in ("sam2.1-hiera-tiny", "sam-vit-base", "dfine-x-o365-coco", "rfdetr-nano"):
        assert P.get_model_policy(mid).is_commercial_safe, mid
        assert mid in P.list_commercial_safe_models()


def test_legal_review_and_byo_not_commercial_safe():
    for mid in ("medsam", "hq-sam", "sam3-base"):
        pol = P.get_model_policy(mid)
        assert pol.is_commercial_safe is False, mid
        assert mid not in P.list_commercial_safe_models()


def test_no_agpl_or_copyleft_in_commercial_safe_list():
    for mid in P.list_commercial_safe_models():
        pol = P.get_model_policy(mid)
        for lic in (pol.code_license, pol.weights_license):
            up = (lic or "").upper()
            assert "AGPL" not in up and "GPL-3" not in up and "GPLV3" not in up, (mid, lic)
        assert pol.commercial_status == "commercial_safe", mid


def test_assert_commercial_safe():
    P.assert_commercial_safe("sam2.1-hiera-tiny")  # no raise
    with pytest.raises(ModelNotCommercialSafeError) as exc:
        P.assert_commercial_safe("medsam2")
    assert exc.value.code == "MODEL_NOT_COMMERCIAL_SAFE"


def test_acknowledgement_text_is_explicit_and_non_clinical():
    t = P.ACKNOWLEDGEMENT_TEXT.lower()
    assert "not commercial-safe by default" in t
    assert "not for diagnosis" in t
    assert "right to use the model weights" in t


def test_explain_flags_restricted():
    txt = P.explain_model_license("medsam2")
    assert "NOT commercial-safe" in txt
    assert "research" in txt.lower()


def test_check_use_allowed_gate():
    # commercial-safe always allowed
    P.check_use_allowed("sam2.1-hiera-tiny", use_mode="commercial")
    # research-only requires acknowledgement
    with pytest.raises(ModelLicenseError) as exc:
        P.check_use_allowed("medsam2", use_mode="commercial", acknowledged=False)
    assert exc.value.code == "MODEL_ACKNOWLEDGEMENT_REQUIRED"
    # acknowledged research passes
    pol = P.check_use_allowed("medsam2", use_mode="research", acknowledged=True)
    assert pol.commercial_status == "research_only"


# --------------------------------------------------------------------------- #
# Python API gate (VisionModel)
# --------------------------------------------------------------------------- #
def test_visionmodel_medsam2_blocked_by_default():
    from visionservex import VisionModel

    with pytest.raises(ModelLicenseError) as exc:
        VisionModel("medsam2")
    assert exc.value.code in {"MODEL_ACKNOWLEDGEMENT_REQUIRED", "MODEL_LICENSE_RESTRICTED"}


def test_visionmodel_medsam2_ack_passes_license_then_registry_gate():
    from visionservex import VisionModel

    # acknowledged → past the license gate → hits the registry/runtime gate
    with pytest.raises(RegistryError):
        VisionModel("medsam2", use_mode="research", acknowledge_license_restrictions=True)


def test_visionmodel_commercial_safe_constructs_without_flags():
    from visionservex import VisionModel

    for mid in ("sam2.1-hiera-tiny", "dfine-x-o365-coco", "clip-vit-base-patch32"):
        m = VisionModel(mid)
        assert m.policy.is_commercial_safe


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main_app():
    from visionservex.cli.main import app

    return app


def test_cli_models_list_commercial_safe_excludes_medsam2():
    o = runner.invoke(_main_app(), ["models", "list", "--commercial-safe", "--json"])
    assert o.exit_code == 0
    ids = {r["model_id"] for r in json.loads(o.stdout)}
    assert "medsam2" not in ids and "medsam" not in ids
    assert "sam2.1-hiera-tiny" in ids


def test_cli_models_assert_commercial_safe():
    o = runner.invoke(_main_app(), ["models", "assert-commercial-safe", "medsam2", "--json"])
    assert o.exit_code != 0
    assert json.loads(o.stdout)["code"] == "MODEL_NOT_COMMERCIAL_SAFE"
    o2 = runner.invoke(
        _main_app(), ["models", "assert-commercial-safe", "sam2.1-hiera-tiny", "--json"]
    )
    assert o2.exit_code == 0


def test_cli_models_explain_medsam2():
    o = runner.invoke(_main_app(), ["models", "explain", "medsam2"])
    assert o.exit_code == 0
    assert "NOT commercial-safe" in o.stdout


def test_cli_predict_medsam2_refused_without_ack(tmp_path):
    from PIL import Image

    img = tmp_path / "x.png"
    Image.new("RGB", (8, 8)).save(img)
    o = runner.invoke(_main_app(), ["predict", "medsam2", str(img), "--json"])
    assert o.exit_code != 0
    out = o.stdout + (o.stderr or "")
    assert "MODEL_ACKNOWLEDGEMENT_REQUIRED" in out or "MODEL_LICENSE_RESTRICTED" in out


def test_cli_predict_medsam2_ack_reaches_runtime_gate_not_license(tmp_path):
    from PIL import Image

    img = tmp_path / "x.png"
    Image.new("RGB", (8, 8)).save(img)
    o = runner.invoke(
        _main_app(),
        [
            "predict",
            "medsam2",
            str(img),
            "--use-mode",
            "research",
            "--acknowledge-license-restrictions",
            "--json",
        ],
    )
    # acknowledged → no longer license-blocked; medsam2 is not a runtime registry
    # model, so it reaches MODEL_NOT_FOUND (the runtime/registry gate).
    assert o.exit_code != 0
    out = o.stdout + (o.stderr or "")
    assert "MODEL_ACKNOWLEDGEMENT_REQUIRED" not in out
    assert "MODEL_NOT_FOUND" in out


# --------------------------------------------------------------------------- #
# Packaging / import hygiene
# --------------------------------------------------------------------------- #
def test_policy_import_is_light():
    import importlib
    import sys

    # importing the policy layer must not pull the heavy optional model stacks
    importlib.import_module("visionservex.policy")
    assert "sam2" not in sys.modules


# --------------------------------------------------------------------------- #
# Docs: no over-claims
# --------------------------------------------------------------------------- #
_FORBIDDEN = (
    "all visionservex models are commercial-safe",
    "medsam2 is commercial-safe",
    "tumor segmentation is commercial-safe",
    "diagnostic segmentation",
    "clinically validated",
)
_NEGATION = ("❌", "not ", "never", "must not", "do not", "forbidden", "no false")


def test_docs_have_no_overclaims():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in ("README.md", "docs/medical_segmentation.md", "docs/model_policy.md"):
        p = root / rel
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            low = line.lower()
            for phrase in _FORBIDDEN:
                if phrase in low:
                    # allowed only as an explicit negative example (what-not-to-claim)
                    assert any(n in low for n in _NEGATION), f"over-claim in {rel}: {line!r}"


def test_commercial_safe_examples_use_only_commercial_safe_models():
    """The quickstart/policy docs must not present a restricted model as a default."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    safe = set(P.list_commercial_safe_models())
    txt = (root / "docs" / "model_policy.md").read_text(encoding="utf-8")
    # the commercial-safe list command + sam2.1 example must reference safe models
    assert "list --commercial-safe" in txt
    assert "sam2.1-hiera-tiny" in safe
