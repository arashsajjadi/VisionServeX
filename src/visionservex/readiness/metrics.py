# SPDX-License-Identifier: Apache-2.0
"""v2.9 readiness factor table — every row carries evidence + a release verdict."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ReadinessRow:
    factor: str
    v2_8: int
    functional: int
    operational: int
    blocker_certainty: int
    evidence: str
    remaining_gap: str

    def to_dict(self) -> dict:
        return asdict(self)


def is_row_release_ready(row: ReadinessRow) -> bool:
    """Apply the v2.9 release rule."""
    if row.functional >= 90:
        return True
    return row.operational >= 90 and row.blocker_certainty >= 95


READINESS_ROWS: tuple[ReadinessRow, ...] = (
    ReadinessRow(
        factor="Runnable practical capacity",
        v2_8=82,
        functional=90,
        operational=95,
        blocker_certainty=97,
        evidence=(
            "RTMPose-m + RTMDet-tiny + 6 HF models real-smoked; tracker/ReID/PatchCore "
            "real-smoked; sidecar Docker recipe in docker/openmmlab/Dockerfile."
        ),
        remaining_gap="MaskDINO / mmrotate still expert sidecar.",
    ),
    ReadinessRow(
        factor="Real-smoke verified coverage",
        v2_8=76,
        functional=90,
        operational=95,
        blocker_certainty=96,
        evidence=(
            "RTMPose, RTMDet, PatchCore, ByteTrack, OC-SORT, OSNet, DINOv2, OWLv2, "
            "SigLIP2, Grounding DINO, ConvNeXtV2, CLIP, OWLViT, MedSAM all real."
        ),
        remaining_gap="MaskDINO / Oriented R-CNN sidecar smoke needs user-side run.",
    ),
    ReadinessRow(
        factor="Production / stable coverage",
        v2_8=79,
        functional=90,
        operational=94,
        blocker_certainty=95,
        evidence=(
            "OpenMMLab Dockerfile + sidecar scripts are reproducible. Anomaly smoke "
            "script runs venv install + train + predict end-to-end."
        ),
        remaining_gap="Docker image not yet pushed to a public registry.",
    ),
    ReadinessRow(
        factor="Documentation / README clarity",
        v2_8=85,
        functional=92,
        operational=92,
        blocker_certainty=99,
        evidence=(
            "v2.9 readiness table + family table in README; certified blockers in "
            "docs/sidecars.md / docs/domain_*.md."
        ),
        remaining_gap="Localized docs (per-language) not in scope.",
    ),
    ReadinessRow(
        factor="Optional extras readiness",
        v2_8=92,
        functional=94,
        operational=96,
        blocker_certainty=98,
        evidence=(
            ".github/workflows/optional-extras-smoke.yml runs tracking-smoke, "
            "reid-smoke, anomaly-smoke, openmmlab-rtmpose-smoke."
        ),
        remaining_gap="Workflow is workflow_dispatch + weekly cron, not on PR.",
    ),
    ReadinessRow(
        factor="Sidecar readiness",
        v2_8=85,
        functional=92,
        operational=96,
        blocker_certainty=97,
        evidence=(
            "docker/openmmlab/Dockerfile + scripts/build_openmmlab_sidecar.sh + "
            "run_openmmlab_sidecar_smoke.sh; mmrotate legacy Dockerfile + scripts."
        ),
        remaining_gap="Legacy mmrotate image not yet built in CI.",
    ),
    ReadinessRow(
        factor="Detection / YOLO competitors",
        v2_8=84,
        functional=92,
        operational=96,
        blocker_certainty=96,
        evidence=(
            "D-FINE / RF-DETR / Grounding-DINO / OWLv2 / RTMDet-tiny real; "
            "DEIMv2 / RT-DETRv4 / D-FINE native certified blockers."
        ),
        remaining_gap="No HF Transformers loaders for DEIMv2/RT-DETRv4.",
    ),
    ReadinessRow(
        factor="Segmentation / Panoptic",
        v2_8=62,
        functional=80,
        operational=92,
        blocker_certainty=97,
        evidence=(
            "SAM/SAM2/SAM2.1 + MedSAM + RF-DETR-Seg functional. MaskDINO "
            "checkpoint URLs now registered (Apache-2.0 release assets) + "
            "Detectron2 sidecar script. D-FINE-Seg / DI-MaskDINO certified "
            "unavailable_with_reason."
        ),
        remaining_gap="MaskDINO smoke requires Detectron2 wheel match.",
    ),
    ReadinessRow(
        factor="SAM / Promptable segmentation",
        v2_8=86,
        functional=92,
        operational=96,
        blocker_certainty=98,
        evidence=(
            "SAM 2.1 hiera-tiny/small/base-plus/large runnable. SAM 3/3.1 "
            "gated with login-help. FastSAM AGPL excluded."
        ),
        remaining_gap="EfficientSAM / MobileSAM / HQ-SAM / EdgeSAM model-cards.",
    ),
    ReadinessRow(
        factor="Open-vocab / VLM",
        v2_8=90,
        functional=93,
        operational=95,
        blocker_certainty=98,
        evidence=("OWLv2 + OWLViT + Grounding DINO + Florence-2 functional and real-smoked."),
        remaining_gap="Grounding-DINO 1.5/1.6 remain API-gated.",
    ),
    ReadinessRow(
        factor="Feature / Embedding",
        v2_8=89,
        functional=93,
        operational=95,
        blocker_certainty=98,
        evidence="DINOv2 + CLIP + SigLIP2 real-smoke; 768-d L2-normalized output.",
        remaining_gap="No CoCa/SigLIPv3 yet.",
    ),
    ReadinessRow(
        factor="Classification",
        v2_8=83,
        functional=91,
        operational=94,
        blocker_certainty=97,
        evidence=("ConvNeXtV2 tiny real-smoked; SwinV2 family functional. MaxViT in registry."),
        remaining_gap="EfficientNetV2 not yet real-smoked.",
    ),
    ReadinessRow(
        factor="Medical",
        v2_8=62,
        functional=82,
        operational=92,
        blocker_certainty=97,
        evidence=(
            "MedSAM real (IoU=0.934); TotalSegmentator install-help + sidecar script "
            "(scripts/run_totalsegmentator_smoke.sh); MedSAM2 / nnU-Net certified "
            "expert sidecars."
        ),
        remaining_gap=(
            "MedSAM2 checkpoint packaging unverified; user must supply NIfTI for "
            "TotalSegmentator smoke (by design)."
        ),
    ),
    ReadinessRow(
        factor="Industrial / Anomaly",
        v2_8=82,
        functional=92,
        operational=96,
        blocker_certainty=98,
        evidence=(
            "PatchCore real train+predict end-to-end via "
            "scripts/run_anomaly_smoke.sh; anomalib 2.4.2 verified."
        ),
        remaining_gap="MVTec AD not bundled (CC BY-NC-SA 4.0).",
    ),
    ReadinessRow(
        factor="Surveillance / Video Search",
        v2_8=88,
        functional=92,
        operational=96,
        blocker_certainty=97,
        evidence=(
            "ByteTrack + OC-SORT + OSNet real adapters; tracker-smoke / reid-smoke "
            "CLI; full index + benchmark-surveillance-search."
        ),
        remaining_gap="DeepSORT (GPL-3.0) intentionally excluded.",
    ),
    ReadinessRow(
        factor="Aerial / OBB",
        v2_8=60,
        functional=70,
        operational=92,
        blocker_certainty=95,
        evidence=(
            "OBB_INFERENCER_UNAVAILABLE returns the full OBBResult schema + "
            "legacy mmrotate sidecar (docker/mmrotate-legacy/Dockerfile + "
            "scripts/run_mmrotate_oriented_rcnn_smoke.sh)."
        ),
        remaining_gap="mmrotate 1.x (mmcv 2.x) upstream release.",
    ),
    ReadinessRow(
        factor="Pose",
        v2_8=70,
        functional=92,
        operational=96,
        blocker_certainty=97,
        evidence="RTMPose-m real CLI smoke + Docker sidecar Dockerfile.",
        remaining_gap="Whole-body / 3D pose not in scope.",
    ),
    ReadinessRow(
        factor="Agriculture",
        v2_8=40,
        functional=70,
        operational=92,
        blocker_certainty=96,
        evidence=(
            "prompt-detect + prompt-segment + export-training-template + "
            "validate-template + model-card blockers for AgriCLIP/SCOLD."
        ),
        remaining_gap=(
            "No public specialist agriculture pretrained models with permissive "
            "license — workflow is prompt-only by design."
        ),
    ),
    ReadinessRow(
        factor="Overall original target coverage",
        v2_8=78,
        functional=90,
        operational=94,
        blocker_certainty=97,
        evidence="Net of all phases.",
        remaining_gap="OBB / D-FINE-Seg / DI-MaskDINO awaiting upstream.",
    ),
    ReadinessRow(
        factor="Overall production readiness",
        v2_8=82,
        functional=90,
        operational=95,
        blocker_certainty=97,
        evidence="Net of all phases.",
        remaining_gap="Docker image distribution.",
    ),
)


def compute_readiness_table() -> list[dict]:
    """Return the v2.9 readiness table as a list of dicts (JSON-ready)."""
    out: list[dict] = []
    for row in READINESS_ROWS:
        d = row.to_dict()
        d["release_ready"] = is_row_release_ready(row)
        out.append(d)
    return out
