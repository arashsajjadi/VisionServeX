# SPDX-License-Identifier: Apache-2.0
"""Tests for v2.6.0: OC-SORT routing, OSNet/Torchreid ReID, OpenMMLab model-card,
MaskDINO sidecar, medical license tiers, SAM3 login-help, license risk table."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

# ---------------------------------------------------------------------------
# Trackers — registry shape + structured errors
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_tracker_list_registry_includes_ocsort():
    from visionservex.runtime.trackers import list_trackers

    reg = list_trackers()
    assert "simple-iou" in reg
    assert "bytetrack" in reg
    assert "ocsort" in reg
    assert reg["simple-iou"]["core_safe"] is True
    assert reg["ocsort"]["license"] == "MIT"


@pytest.mark.fast
def test_tracker_list_excludes_gpl_from_permissive_core():
    from visionservex.runtime.trackers import list_trackers

    reg = list_trackers()
    assert reg["deepsort"]["core_safe"] is False
    assert reg["deepsort"]["license"].startswith("GPL")


@pytest.mark.fast
def test_build_tracker_ocsort_missing():
    from visionservex.runtime.trackers import TrackerUnavailableError, build_tracker

    with pytest.raises(TrackerUnavailableError) as exc:
        build_tracker("ocsort")
    payload = exc.value.to_dict()
    assert payload["code"] == "OCSORT_REQUIRED"
    assert "pip install ocsort" in payload["install"]


@pytest.mark.fast
def test_build_tracker_deepsort_is_blocked_in_core():
    from visionservex.runtime.trackers import TrackerUnavailableError, build_tracker

    with pytest.raises(TrackerUnavailableError) as exc:
        build_tracker("deepsort")
    payload = exc.value.to_dict()
    assert payload["code"] == "DEEPSORT_GPL_BLOCKED"


@pytest.mark.fast
def test_ocsort_adapter_uses_mocked_package(monkeypatch):
    """Simulate ocsort being installed and verify update() converts rows correctly."""
    import numpy as np

    fake_module = MagicMock()
    fake_sub = MagicMock()

    class _FakeOCSort:
        def __init__(self, **_):
            pass

        def update(self, dets, img_info=None, img_size=None):
            # Return [x1, y1, x2, y2, track_id] rows matching dets.
            out = []
            for row in dets:
                out.append([row[0], row[1], row[2], row[3], 7])
            return np.asarray(out, dtype=float)

    fake_module.OCSort = _FakeOCSort
    fake_sub.OCSort = _FakeOCSort
    monkeypatch.setitem(sys.modules, "ocsort", fake_module)
    monkeypatch.setitem(sys.modules, "ocsort.ocsort", fake_sub)

    from visionservex.runtime.trackers import build_tracker

    tracker = build_tracker("ocsort")
    detections = [((10.0, 20.0, 100.0, 200.0), 0.9, "person")]
    out = tracker.update(detections, frame_idx=0, timestamp_s=0.5, img_size=(640, 480))

    assert len(out) == 1
    tb = out[0]
    assert tb.track_id == 7
    assert tb.label == "person"
    assert tb.box == (10.0, 20.0, 100.0, 200.0)
    assert tb.frame_idx == 0
    assert tb.timestamp_s == 0.5


# ---------------------------------------------------------------------------
# ReID — Torchreid/OSNet adapter
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_list_reid_includes_osnet():
    from visionservex.runtime.reid import list_reid

    reg = list_reid()
    assert "osnet" in reg
    assert reg["osnet"]["license"] == "MIT"
    assert reg["osnet"]["core_safe"] is True
    assert reg["fastreid"]["core_safe"] is False  # expert sidecar


@pytest.mark.fast
def test_build_reid_extractor_missing_torchreid():
    import importlib.util

    from visionservex.runtime.reid import ReIDUnavailableError, build_reid_extractor

    with pytest.raises(ReIDUnavailableError) as exc:
        build_reid_extractor("osnet")
    payload = exc.value.to_dict()
    if importlib.util.find_spec("torchreid") is not None:
        # torchreid present -> import check passes; the next blocker is the missing checkpoint.
        assert payload["code"] == "REID_CHECKPOINT_REQUIRED"
    else:
        assert payload["code"] in {"TORCHREID_REQUIRED", "REID_UNAVAILABLE"}
        assert (
            "torchreid" in payload["install"].lower()
            or "deep-person-reid" in payload["install"]
        )


@pytest.mark.fast
def test_build_reid_extractor_checkpoint_required(monkeypatch, tmp_path):
    """torchreid installed but model_path missing should yield CHECKPOINT_REQUIRED."""
    fake = MagicMock()
    fake.utils = MagicMock()

    class _FakeFeatureExtractor:
        def __init__(self, **_):
            pass

    fake.utils.FeatureExtractor = _FakeFeatureExtractor
    monkeypatch.setitem(sys.modules, "torchreid", fake)
    monkeypatch.setitem(sys.modules, "torchreid.utils", fake.utils)

    from visionservex.runtime.reid import ReIDUnavailableError, build_reid_extractor

    with pytest.raises(ReIDUnavailableError) as exc:
        build_reid_extractor("osnet", model_path=None)
    assert exc.value.to_dict()["code"] == "REID_CHECKPOINT_REQUIRED"


@pytest.mark.fast
def test_torchreid_osnet_adapter_extract_normalizes(monkeypatch, tmp_path):
    """When torchreid + checkpoint present, extract() returns normalized embeddings."""
    import numpy as np

    fake = MagicMock()
    fake.utils = MagicMock()

    class _FakeFeatureExtractor:
        def __init__(self, **_):
            pass

        def __call__(self, _images):
            arr = np.array([[3.0, 4.0]])  # L2-norm = 5
            tensor = MagicMock()
            tensor.detach.return_value.cpu.return_value.numpy.return_value = arr
            return tensor

    fake.utils.FeatureExtractor = _FakeFeatureExtractor
    monkeypatch.setitem(sys.modules, "torchreid", fake)
    monkeypatch.setitem(sys.modules, "torchreid.utils", fake.utils)

    ckpt = tmp_path / "osnet.pth"
    ckpt.write_bytes(b"fake")

    from visionservex.runtime.reid import build_reid_extractor

    extractor = build_reid_extractor("osnet", model_path=str(ckpt))
    out = extractor.extract([object()])
    assert out.shape == (1, 2)
    # L2-normalized: (3,4)/5 = (0.6, 0.8)
    assert abs(out[0, 0] - 0.6) < 1e-6
    assert abs(out[0, 1] - 0.8) < 1e-6


# ---------------------------------------------------------------------------
# video-search CLI — reid-smoke
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_video_search_reid_smoke_torchreid_required(tmp_path):
    from visionservex.cli.video_search_commands import app as vs_app

    # Need an image file to pass the INPUT_NOT_FOUND check.
    img = tmp_path / "person.jpg"
    img.write_bytes(b"not_really_an_image")

    runner = CliRunner()
    result = runner.invoke(vs_app, ["reid-smoke", "--image", str(img), "--json"])
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["code"] in {"TORCHREID_REQUIRED", "REID_CHECKPOINT_REQUIRED"}


@pytest.mark.fast
def test_video_search_reid_smoke_missing_image():
    from visionservex.cli.video_search_commands import app as vs_app

    runner = CliRunner()
    result = runner.invoke(
        vs_app,
        ["reid-smoke", "--image", "/tmp/does_not_exist_12345.jpg", "--json"],
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["code"] == "INPUT_NOT_FOUND"


# ---------------------------------------------------------------------------
# OpenMMLab — model-card metadata
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_openmmlab_model_card_rtmdet_l_coco():
    from visionservex.cli.openmmlab_commands import app as oml_app

    runner = CliRunner()
    result = runner.invoke(oml_app, ["model-card", "rtmdet-l-coco", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["task"] == "detect"
    assert payload["license"] == "Apache-2.0"
    assert payload["inferencer"] == "mmdet.apis.DetInferencer"
    assert "download.openmmlab.com" in payload["download_url"]
    assert payload["checkpoint_filename"].endswith(".pth")


@pytest.mark.fast
def test_openmmlab_model_card_rtmpose_m():
    from visionservex.cli.openmmlab_commands import app as oml_app

    runner = CliRunner()
    result = runner.invoke(oml_app, ["model-card", "rtmpose-m", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["task"] == "pose"
    assert payload["inferencer"] == "mmpose.apis.MMPoseInferencer"
    assert "rtmpose-m" in payload["checkpoint_filename"]


@pytest.mark.fast
def test_openmmlab_oriented_rcnn_obb_metadata():
    from visionservex.cli.openmmlab_commands import app as oml_app

    runner = CliRunner()
    result = runner.invoke(oml_app, ["model-card", "oriented-rcnn", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["task"] == "obb"
    # Must not flatten OBB to xyxy.
    assert "theta" in payload["note"].lower()


@pytest.mark.fast
def test_openmmlab_validate_missing_deps_returns_structured_error():
    from visionservex.cli.openmmlab_commands import app as oml_app

    runner = CliRunner()
    result = runner.invoke(oml_app, ["validate", "rtmdet-l-coco", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["structured_error_code"] in {"OPENMMLAB_REQUIRED", "CHECKPOINT_REQUIRED"}


@pytest.mark.fast
def test_openmmlab_model_card_unknown():
    from visionservex.cli.openmmlab_commands import app as oml_app

    runner = CliRunner()
    result = runner.invoke(oml_app, ["model-card", "not-real-model", "--json"])
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["code"] == "CONFIG_REQUIRED"


# ---------------------------------------------------------------------------
# MaskDINO sidecar
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_maskdino_doctor_returns_detectron2_required():
    from visionservex.cli.maskdino_commands import app as md_app

    runner = CliRunner()
    result = runner.invoke(md_app, ["doctor", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["detectron2_installed"] is False
    assert payload["code"] == "DETECTRON2_REQUIRED"


@pytest.mark.fast
def test_maskdino_validate_swinl_returns_structured_blocker():
    from visionservex.cli.maskdino_commands import app as md_app

    runner = CliRunner()
    result = runner.invoke(md_app, ["validate", "maskdino-swinl-coco", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["structured_error_code"] in {"DETECTRON2_REQUIRED", "CHECKPOINT_REQUIRED"}
    assert payload["license"] == "Apache-2.0"
    # v2.9: checkpoint URL is the OFFICIAL IDEA-Research/detrex-storage URL.
    # In v2.6 it was correctly None (URL not in research set); the upstream
    # README later resolved this, so the URL is now present and verified.
    if payload.get("checkpoint_url") is not None:
        assert payload["checkpoint_url"].startswith(
            "https://github.com/IDEA-Research/detrex-storage/releases/download/"
        ), "if checkpoint_url is present it must point at the official release tag"


@pytest.mark.fast
def test_maskdino_smoke_test_returns_structured_blocker():
    from visionservex.cli.maskdino_commands import app as md_app

    runner = CliRunner()
    result = runner.invoke(
        md_app,
        ["smoke-test", "maskdino-swinl-coco", "/tmp/anything.jpg", "--json"],
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["structured_error_code"] == "DETECTRON2_REQUIRED"


@pytest.mark.fast
def test_maskdino_create_env_recipe_pinned():
    from visionservex.cli.maskdino_commands import app as md_app

    runner = CliRunner()
    result = runner.invoke(md_app, ["create-env", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["env_name"]
    assert any("detectron2" in cmd for cmd in payload["commands"])
    assert any("MaskDINO" in cmd for cmd in payload["commands"])
    assert payload["license"] == "Apache-2.0"


# ---------------------------------------------------------------------------
# Medical — license tier + install-help + monai + autoseg
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_medical_install_help_totalsegmentator_tissue_marked_non_core():
    from visionservex.cli.medical_commands import app as med_app

    runner = CliRunner()
    result = runner.invoke(med_app, ["install-help", "totalsegmentator-tissue", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    entry = payload["totalsegmentator-tissue"]
    assert entry["license_tier"] == "non_core_license_optional"
    assert entry["structured_error_code"] == "TOTALSEGMENTATOR_LICENSE_REQUIRED"


@pytest.mark.fast
def test_medical_install_help_medsam2_marked_expert_sidecar():
    from visionservex.cli.medical_commands import app as med_app

    runner = CliRunner()
    result = runner.invoke(med_app, ["install-help", "medsam2", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    entry = payload["medsam2"]
    assert entry["license_tier"] == "expert_sidecar"
    assert entry["checkpoint_status"] == "unverified"


@pytest.mark.fast
def test_medical_autoseg_doctor_monai_required():
    from visionservex.cli.medical_commands import app as med_app

    runner = CliRunner()
    result = runner.invoke(med_app, ["autoseg", "doctor", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    if not payload.get("monai_installed", False):
        assert payload["code"] == "AUTO3DSEG_REQUIRED"


@pytest.mark.fast
def test_medical_monai_list_bundles_returns_blocker_when_missing():
    from visionservex.cli.medical_commands import app as med_app

    runner = CliRunner()
    result = runner.invoke(med_app, ["monai", "list-bundles", "--json"])
    payload = json.loads(result.stdout)
    if result.exit_code != 0:
        assert payload["code"] == "MONAI_REQUIRED"


# ---------------------------------------------------------------------------
# SAM family — login-help + validate
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_sam_family_login_help_sam31_returns_gated_status():
    from visionservex.cli.sam_family_commands import app as sf_app

    runner = CliRunner()
    result = runner.invoke(sf_app, ["login-help", "sam3.1", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["code"] == "GATED_HF_AUTH_REQUIRED"
    assert "huggingface-cli login" in " ".join(payload["steps"])


@pytest.mark.fast
def test_sam_family_validate_sam3_base_emits_gated_status():
    from visionservex.cli.sam_family_commands import app as sf_app

    runner = CliRunner()
    result = runner.invoke(sf_app, ["validate", "sam3-base", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["code"] == "GATED_HF_AUTH_REQUIRED"
    assert payload["family"] == "sam3"


# ---------------------------------------------------------------------------
# License risk table — file + manifest invariants
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_license_risk_table_exists():
    p = Path(__file__).resolve().parents[1] / "docs" / "license_risk_table.md"
    assert p.exists()
    body = p.read_text()
    assert "Core-safe" in body
    assert "Non-core / excluded" in body
    assert "AGPL-3.0" in body
    assert "GPL-3.0" in body
    assert "PML 1.0" in body


@pytest.mark.fast
def test_manifest_deepsort_not_runnable_in_permissive_core():
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST

    for entry in SOURCE_MANIFEST.values():
        if entry.license.lower().startswith("gpl"):
            assert not entry.runnable_in_visionservex, (
                f"{entry.model_id} is GPL but marked runnable"
            )
            assert entry.recommended_action in {
                "do_not_add",
                "audit_only",
                "expert_sidecar",
                "external_api",
                "non_core_license_optional",
                "unavailable_with_reason",
            }, f"{entry.model_id} GPL with action={entry.recommended_action}"


@pytest.mark.fast
def test_manifest_fastsam_excluded_or_sidecar():
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST

    for entry in SOURCE_MANIFEST.values():
        if "fastsam" in entry.model_id.lower():
            assert not entry.runnable_in_visionservex, (
                f"{entry.model_id} (AGPL) must not be runnable in permissive core"
            )


@pytest.mark.fast
def test_manifest_rtdetrv4_carries_paper_url():
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST

    rt = SOURCE_MANIFEST["rtdetrv4-s"]
    assert rt.paper_url.startswith("https://arxiv.org")
    assert "Date checked" in " ".join(rt.known_blockers)


# ---------------------------------------------------------------------------
# Version sanity
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_version_is_at_least_260():
    """Version pin: 2.6.0 introduced these features; v2.7+ must not regress them."""
    import visionservex

    parts = tuple(int(p) for p in visionservex.__version__.split(".")[:3])
    assert parts >= (2, 6, 0), f"version {visionservex.__version__} < 2.6.0"
