# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.23: medical segmentation pack — truth, license-drift, honest sidecar, multi-box.

These tests are weight-free and download-free. They assert that:

* MedSAM v1 is a real, wired, 2D promptable model and now returns ONE mask per box.
* MedSAM2 is an honest, dependency-gated, research-only sidecar that fails cleanly
  and is NEVER labelled commercial-safe (no false runtime claim, no mock output).
* License metadata agrees across the authoritative policy, the model-zoo manifest,
  the registry, and the medical CLI (drift firewall).
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from visionservex.licensing.policy import get_policy
from visionservex.model_zoo import SOURCE_MANIFEST
from visionservex.registry import RegistryError, default_registry

runner = CliRunner()


# --------------------------------------------------------------------------- #
# Registry / policy truth
# --------------------------------------------------------------------------- #
def test_medsam_v1_is_wired_promptable_and_research_only():
    e = default_registry().get("medsam")
    assert e.task == "foundation_segment"
    assert e.engine == "sam_hf"
    assert e.implementation_status == "wired"
    # research-only labelling is now explicit on the registry entry
    assert e.license_uncertain is True
    assert e.commercial_use_notes and "commercial-safe" in e.commercial_use_notes.lower()


def test_medsam_v1_policy_is_legal_review_not_commercial_safe():
    pol = get_policy("medsam")
    assert pol is not None
    assert pol.final_policy == "legal_review_required"
    assert pol.commercial_safe is False
    assert pol.default_safe is False


def test_medsam2_is_not_a_runtime_registry_model():
    """Honesty: MedSAM2 must NOT masquerade as a runtime model in the registry."""
    with pytest.raises(RegistryError):
        default_registry().get("medsam2")


def test_medsam2_policy_is_noncommercial_not_commercial_safe():
    pol = get_policy("medsam2")
    assert pol is not None
    assert pol.final_policy == "noncommercial_restricted"
    assert pol.commercial_safe is False
    assert pol.default_safe is False


# --------------------------------------------------------------------------- #
# License-drift firewall: policy is authoritative; manifest must not contradict.
# --------------------------------------------------------------------------- #
def test_medical_manifest_does_not_contradict_noncommercial_policy():
    """Any medical model the policy says is NOT commercial-safe must surface risk
    in the manifest (so the gap report can never show a bare permissive license)."""
    offenders = []
    for mid, src in SOURCE_MANIFEST.items():
        if src.domain != "medical":
            continue
        pol = get_policy(mid)
        if pol is None or pol.commercial_safe:
            continue
        signals_risk = (src.license_risk != "none") or bool(src.known_blockers)
        if not signals_risk:
            offenders.append(mid)
    assert not offenders, f"medical manifest hides non-commercial/legal-review risk: {offenders}"


def test_medsam2_manifest_row_is_honest():
    src = SOURCE_MANIFEST["medsam2"]
    assert src.runnable_in_visionservex is False
    assert src.license_risk == "non_commercial"
    assert "non-commercial" in src.license.lower()
    assert src.known_blockers, "medsam2 must list blockers so the gap report is honest"


# --------------------------------------------------------------------------- #
# MedSAM2 honest dependency-gated sidecar (no false runtime claim, no mock)
# --------------------------------------------------------------------------- #
def test_medsam2_sidecar_engine_registered():
    from visionservex.engines.registry import _FACTORIES

    assert "medsam2_sidecar" in _FACTORIES


def test_medsam2_probe_reports_unrunnable_noncommercial():
    from visionservex.engines.medsam2_sidecar import probe_medsam2_availability

    p = probe_medsam2_availability()
    assert p["runnable"] is False
    assert p["commercial_safe"] is False
    assert p["runtime_status"] == "expert_sidecar"
    # sam2 is absent in CI -> the structured blocker is the dependency one
    assert "sam2" in p["missing_modules"]
    assert p["structured_error_code"] == "MEDSAM2_REQUIRED"
    assert "non-commercial" in p["license_note"].lower()


def test_medsam2_sidecar_load_raises_structured_missing_dependency():
    from visionservex.engines.base import MissingDependencyError
    from visionservex.engines.medsam2_sidecar import MedSAM2SidecarEngine
    from visionservex.registry import ModelEntry

    entry = ModelEntry(
        id="medsam2",
        display_name="MedSAM2 (research-only sidecar)",
        task="foundation_segment",
        family="medsam2",
        engine="medsam2_sidecar",
        license="Apache-2.0 code / non-commercial weights",
        upstream_url="https://github.com/bowang-lab/MedSAM2",
    )
    eng = MedSAM2SidecarEngine(entry)
    with pytest.raises(MissingDependencyError) as exc:
        eng.load(device="cpu", precision="fp32")
    msg = str(exc.value)
    assert "MedSAM2" in msg
    assert "non-commercial" in msg.lower()
    # honest: it did NOT silently become "ready" with mock output
    assert eng._real_ready is False


# --------------------------------------------------------------------------- #
# MedSAM v1 multi-box correctness (the real bug fix): N boxes -> N masks
# --------------------------------------------------------------------------- #
def _fake_sam_engine(masks, iou):
    """Build a SAMHFEngine with fake model/processor returning the given tensors."""
    import torch

    from visionservex.engines.sam_hf import SAMHFEngine
    from visionservex.registry import default_registry

    eng = SAMHFEngine(default_registry().get("medsam"))
    eng.device = "cpu"
    eng.precision = "fp32"
    eng._torch = torch
    eng._real_ready = True

    n, k, h, w = masks.shape

    class _ImgProc:
        def post_process_masks(self, pred_masks, orig, reshaped):
            return [masks]

    class _Proc:
        def __init__(self):
            self.image_processor = _ImgProc()

        def __call__(self, **kw):
            return {
                "pixel_values": torch.zeros(1, 3, h, w),
                "original_sizes": torch.tensor([[h, w]]),
                "reshaped_input_sizes": torch.tensor([[h, w]]),
            }

    class _Model:
        def __init__(self):
            self._p = torch.zeros(1)

        def parameters(self):
            return iter([self._p])

        def __call__(self, **kw):
            out = type("O", (), {})()
            out.pred_masks = torch.zeros(1, n, k, h, w)
            out.iou_scores = iou
            return out

    eng._processor = _Proc()
    eng._model = _Model()
    return eng


def test_medsam_multibox_returns_one_segment_per_box():
    pytest.importorskip("torch")  # engine-level test needs real tensors
    import numpy as np
    import torch
    from PIL import Image

    h = w = 16
    masks = torch.zeros(2, 1, h, w, dtype=torch.uint8)
    masks[0, 0, :, :8] = 1  # box 0 -> left half
    masks[1, 0, :, 8:] = 1  # box 1 -> right half
    iou = torch.tensor([[[0.9], [0.8]]])  # (1, num_prompts=2, num_masks=1)

    eng = _fake_sam_engine(masks, iou)
    res = eng.predict(Image.new("RGB", (w, h)), boxes=[[0, 0, 8, 16], [8, 0, 16, 16]])

    assert len(res.segments) == 2, "multi-box must yield one mask per box (regression guard)"
    left, right = res.segments[0].mask, res.segments[1].mask
    assert np.array(left)[:, :8].all() and not np.array(left)[:, 8:].any()
    assert np.array(right)[:, 8:].all() and not np.array(right)[:, :8].any()


def test_medsam_single_box_returns_one_segment():
    pytest.importorskip("torch")  # engine-level test needs real tensors
    import torch
    from PIL import Image

    h = w = 16
    masks = torch.zeros(1, 1, h, w, dtype=torch.uint8)
    masks[0, 0, 4:12, 4:12] = 1
    iou = torch.tensor([[[0.95]]])
    eng = _fake_sam_engine(masks, iou)
    res = eng.predict(Image.new("RGB", (w, h)), boxes=[[4, 4, 12, 12]])
    assert len(res.segments) == 1
    assert res.segments[0].score == pytest.approx(0.95, abs=1e-5)


# --------------------------------------------------------------------------- #
# Medical CLI smoke (structured, honest)
# --------------------------------------------------------------------------- #
def _med_app():
    from visionservex.cli.medical_commands import app

    return app


def test_cli_medical_install_help_medsam2_marks_noncommercial_sidecar():
    res = runner.invoke(_med_app(), ["install-help", "medsam2", "--json"])
    assert res.exit_code == 0
    entry = json.loads(res.stdout)["medsam2"]
    assert entry["license_tier"] == "expert_sidecar"
    assert entry["checkpoint_status"] == "unverified"
    assert "non-commercial" in entry["license_note"].lower()


def test_cli_medical_segment_medsam2_emits_structured_blocker(tmp_path):
    img = tmp_path / "x.png"
    from PIL import Image

    Image.new("RGB", (8, 8)).save(img)
    res = runner.invoke(
        _med_app(),
        ["segment", "medsam2", str(img), "--out", str(tmp_path / "out"), "--json"],
    )
    # sam2/medsam2 absent -> structured dependency blocker, non-zero exit, no crash
    assert res.exit_code != 0
    payload = json.loads(res.stdout)
    assert payload["code"] == "MEDSAM2_CHECKPOINT_UNVERIFIED"


def test_cli_medical_list_includes_both_medsam_and_medsam2():
    res = runner.invoke(_med_app(), ["list", "--json"])
    assert res.exit_code == 0
    ids = {row["id"] for row in json.loads(res.stdout)}
    assert {"medsam", "medsam2"} <= ids


# --------------------------------------------------------------------------- #
# HTTP honesty: MedSAM2 is research-only (non-commercial) -> the commercial-safe
# policy applies over HTTP too: a clean 403 license error, never a 500 or a
# fabricated mask. (Also a regression guard for the segment/b64 error path.)
# --------------------------------------------------------------------------- #
def test_http_segment_b64_medsam2_is_license_gated_not_500():
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    import base64
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    from visionservex.config import reload_settings
    from visionservex.server.app import create_app

    app = create_app(reload_settings())
    client = TestClient(app, raise_server_exceptions=False)
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    r = client.post(
        "/segment/b64",
        json={"model_id": "medsam2", "image_b64": b64, "options": {"boxes": [[1, 1, 6, 6]]}},
    )
    # research-only model refused with a structured license error, never a 500
    assert r.status_code == 403
    assert r.json()["error"]["code"] in {
        "MODEL_ACKNOWLEDGEMENT_REQUIRED",
        "MODEL_LICENSE_RESTRICTED",
        "MODEL_NOT_COMMERCIAL_SAFE",
    }
