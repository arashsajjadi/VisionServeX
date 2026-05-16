# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Source-grounded model manifest.

Every model integrated in VisionServeX is cited here with the exact official
URLs from which the integration was derived. This provides traceability and
honest license/access metadata.

Add a new model: append a ModelSource entry below. Mark `runnable_in_visionservex`
honestly — do not claim a model is wired without real inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelSource:
    """Source metadata for a model integration.

    Fields:
        model_id: VisionServeX registry ID (e.g. "dfine-x-o365-coco").
        family: model family (e.g. "dfine", "rfdetr", "dinov2").
        task: VisionServeX task ("detect", "segment", "embed", "vlm", ...).
        official_repo: primary upstream GitHub URL.
        official_docs: documentation/project page URL.
        paper_url: paper/arXiv URL (optional).
        hf_repo: Hugging Face repository ID.
        checkpoint_url: direct checkpoint URL (or empty if N/A).
        checkpoint_source: HF hub / official repo / API / community.
        checkpoint_trust_level: official_upstream | official_hf | community_hf |
            converted | unverified | api_only.
        license: SPDX-like identifier or short text.
        license_risk: none | check | restricted | non_commercial | gpl |
            agpl | api_only | unknown.
        install_command: exact pip command.
        hf_class: Transformers AutoModel* class if HF-based (else "").
        runnable_in_visionservex: True if real inference works in this build.
        access_status: open | hf_login | api_token | gated | manual_download |
            unavailable.
        domain: medical | agriculture | aerial | industrial | surveillance |
            general | open_vocab | feature_backbone (free-form).
        known_blockers: list of exact reasons a model is not yet wired.
        recommended_action: add_now | expert_sidecar | external_api |
            non_core_license_optional | audit_only | do_not_add.
    """

    model_id: str
    family: str
    task: str
    official_repo: str = ""
    official_docs: str = ""
    paper_url: str = ""
    hf_repo: str = ""
    checkpoint_url: str = ""
    checkpoint_source: str = ""
    checkpoint_trust_level: str = "unverified"
    license: str = "Apache-2.0"
    license_risk: str = "none"
    install_command: str = "pip install visionservex"
    hf_class: str = ""
    runnable_in_visionservex: bool = False
    access_status: str = "open"
    domain: str = "general"
    known_blockers: list[str] = field(default_factory=list)
    recommended_action: str = "audit_only"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "task": self.task,
            "official_repo": self.official_repo,
            "official_docs": self.official_docs,
            "paper_url": self.paper_url,
            "hf_repo": self.hf_repo,
            "checkpoint_url": self.checkpoint_url,
            "checkpoint_source": self.checkpoint_source,
            "checkpoint_trust_level": self.checkpoint_trust_level,
            "license": self.license,
            "license_risk": self.license_risk,
            "install_command": self.install_command,
            "hf_class": self.hf_class,
            "runnable_in_visionservex": self.runnable_in_visionservex,
            "access_status": self.access_status,
            "domain": self.domain,
            "known_blockers": self.known_blockers,
            "recommended_action": self.recommended_action,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Source manifest — every URL below is referenced from the integration task.
# ---------------------------------------------------------------------------

SOURCE_MANIFEST: dict[str, ModelSource] = {
    # =====================================================================
    # D-FINE — runnable via HF ustc-community checkpoints (v1.2.0)
    # =====================================================================
    "dfine-s-o365-coco": ModelSource(
        model_id="dfine-s-o365-coco",
        family="dfine",
        task="detect",
        official_repo="https://github.com/Peterande/D-FINE",
        hf_repo="ustc-community/dfine-small-obj2coco",
        checkpoint_source="hf_hub",
        checkpoint_trust_level="community_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="AutoModelForObjectDetection",
        runnable_in_visionservex=True,
        access_status="open",
        domain="general",
        recommended_action="add_now",
        notes="HF community checkpoint. Official .pth not yet integrated.",
    ),
    "dfine-x-o365-coco": ModelSource(
        model_id="dfine-x-o365-coco",
        family="dfine",
        task="detect",
        official_repo="https://github.com/Peterande/D-FINE",
        hf_repo="ustc-community/dfine-xlarge-obj2coco",
        checkpoint_source="hf_hub",
        checkpoint_trust_level="community_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="AutoModelForObjectDetection",
        runnable_in_visionservex=True,
        access_status="open",
        domain="general",
        recommended_action="add_now",
        notes="Strongest D-FINE in current build. YOLO26-X comparison target.",
    ),
    "dfine-l-o365-coco": ModelSource(
        model_id="dfine-l-o365-coco",
        family="dfine",
        task="detect",
        official_repo="https://github.com/Peterande/D-FINE",
        hf_repo="ustc-community/dfine-large-obj2coco-e25",
        checkpoint_source="hf_hub",
        checkpoint_trust_level="community_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="AutoModelForObjectDetection",
        runnable_in_visionservex=True,
        access_status="open",
        domain="general",
        recommended_action="add_now",
    ),
    # =====================================================================
    # RF-DETR — Apache-2.0 sizes only in core
    # =====================================================================
    "rfdetr-small": ModelSource(
        model_id="rfdetr-small",
        family="rfdetr",
        task="detect",
        official_repo="https://github.com/roboflow/rf-detr",
        official_docs="https://rfdetr.roboflow.com/",
        checkpoint_source="package_managed",
        checkpoint_trust_level="official_upstream",
        license="Apache-2.0",
        install_command="pip install 'visionservex[rfdetr]'",
        runnable_in_visionservex=True,
        access_status="open",
        domain="general",
        recommended_action="add_now",
    ),
    "rfdetr-large": ModelSource(
        model_id="rfdetr-large",
        family="rfdetr",
        task="detect",
        official_repo="https://github.com/roboflow/rf-detr",
        official_docs="https://rfdetr.roboflow.com/",
        checkpoint_source="package_managed",
        checkpoint_trust_level="official_upstream",
        license="Apache-2.0",
        install_command="pip install 'visionservex[rfdetr]'",
        runnable_in_visionservex=True,
        access_status="open",
        domain="general",
        recommended_action="add_now",
        notes="YOLO26-X competitor candidate.",
    ),
    "rfdetr-seg-medium": ModelSource(
        model_id="rfdetr-seg-medium",
        family="rfdetr",
        task="segment",
        official_repo="https://github.com/roboflow/rf-detr",
        official_docs="https://rfdetr.roboflow.com/develop/reference/seg_large/",
        checkpoint_source="package_managed",
        checkpoint_trust_level="official_upstream",
        license="Apache-2.0",
        install_command="pip install 'visionservex[rfdetr]'",
        runnable_in_visionservex=True,
        access_status="open",
        domain="general",
        recommended_action="add_now",
    ),
    "rfdetr-seg-large": ModelSource(
        model_id="rfdetr-seg-large",
        family="rfdetr",
        task="segment",
        official_repo="https://github.com/roboflow/rf-detr",
        official_docs="https://rfdetr.roboflow.com/develop/reference/seg_large/",
        checkpoint_source="package_managed",
        license="PML 1.0 (Plus/XL/2XL); core sizes Apache-2.0",
        license_risk="restricted",
        runnable_in_visionservex=False,
        access_status="open",
        domain="general",
        known_blockers=[
            "Plus / XL / 2XL variants use the Roboflow PML 1.0 license — restrictive.",
            "Must remain non_core_license_optional; do not auto-pull by default.",
            "Date checked: 2026-05-16.",
        ],
        recommended_action="non_core_license_optional",
        notes=(
            "Install via pip install rfdetr[plus]; users must accept PML 1.0 terms. Core"
            " Apache-2.0 sizes (small/base/large) remain runnable through other entries."
        ),
    ),
    # =====================================================================
    # DEIMv2 — experimental, not yet wired
    # =====================================================================
    "deimv2-s": ModelSource(
        model_id="deimv2-s",
        family="deimv2",
        task="detect",
        official_repo="https://github.com/Intellindust-AI-Lab/DEIMv2",
        official_docs="https://intellindust-ai-lab.github.io/projects/DEIMv2/",
        paper_url="https://huggingface.co/papers/2509.20787",
        hf_repo="Intellindust/DEIMv2_DINOv3_S_COCO",
        checkpoint_source="hf_hub_candidate",
        checkpoint_trust_level="unverified",
        license="Apache-2.0",
        license_risk="check",
        install_command="git clone https://github.com/Intellindust-AI-Lab/DEIMv2 (manual)",
        runnable_in_visionservex=False,
        access_status="open",
        domain="general",
        known_blockers=[
            "Not yet supported by HF Transformers (issue #41211).",
            "Requires native PyTorch loader from official repo.",
            "HF candidate model name not verified live.",
        ],
        recommended_action="audit_only",
        notes="experimental_sota. DINOv3 backbone variant from project page.",
    ),
    "deimv2-m": ModelSource(
        model_id="deimv2-m",
        family="deimv2",
        task="detect",
        official_repo="https://github.com/Intellindust-AI-Lab/DEIMv2",
        runnable_in_visionservex=False,
        known_blockers=["Same as deimv2-s — repo native loader required."],
        recommended_action="audit_only",
    ),
    "deimv2-l": ModelSource(
        model_id="deimv2-l",
        family="deimv2",
        task="detect",
        official_repo="https://github.com/Intellindust-AI-Lab/DEIMv2",
        runnable_in_visionservex=False,
        known_blockers=["Heavy model. Native loader required."],
        recommended_action="audit_only",
    ),
    "deimv2-x": ModelSource(
        model_id="deimv2-x",
        family="deimv2",
        task="detect",
        official_repo="https://github.com/Intellindust-AI-Lab/DEIMv2",
        runnable_in_visionservex=False,
        known_blockers=["Heaviest DEIMv2. Native loader required."],
        recommended_action="audit_only",
    ),
    # =====================================================================
    # RT-DETRv4 — experimental, unverified
    # =====================================================================
    "rtdetrv4-s": ModelSource(
        model_id="rtdetrv4-s",
        family="rtdetrv4",
        task="detect",
        official_repo="https://github.com/RT-DETRs/RT-DETRv4",
        paper_url="https://arxiv.org/abs/2510.25257",
        license="Apache-2.0",
        license_risk="none",
        runnable_in_visionservex=False,
        access_status="open",
        known_blockers=[
            "RT-DETRv4 native loader requires repo-internal config/checkpoint mapping.",
            "No clean HF/native pip loader — install via git clone + requirements.txt.",
            "Known TensorRT/RTX 5080 issues reported in upstream issues.",
            "Date checked: 2026-05-16.",
        ],
        recommended_action="expert_sidecar",
        notes=(
            "Run upstream tools/inference/torch_inf.py with config + checkpoint from the official"
            " release. Apache-2.0; license_risk lowered to none after re-verification."
        ),
    ),
    # =====================================================================
    # Co-DINO / Co-DETR — expert sidecar
    # =====================================================================
    "co-dino-inst-vit-l-coco": ModelSource(
        model_id="co-dino-inst-vit-l-coco",
        family="co-dino",
        task="segment",
        official_repo="https://github.com/Sense-X/Co-DETR",
        license="Apache-2.0",
        license_risk="check",
        runnable_in_visionservex=False,
        access_status="open",
        known_blockers=[
            "Requires OpenMMLab MMDetection sidecar.",
            "OpenMMLab not installed by default in core.",
        ],
        recommended_action="expert_sidecar",
        notes="Expert path via visionservex openmmlab pull/smoke-test.",
    ),
    # =====================================================================
    # MaskDINO — Detectron2 sidecar
    # =====================================================================
    "maskdino-swinl-coco": ModelSource(
        model_id="maskdino-swinl-coco",
        family="maskdino",
        task="segment",
        official_repo="https://github.com/IDEA-Research/MaskDINO",
        license="Apache-2.0",
        license_risk="check",
        runnable_in_visionservex=False,
        access_status="open",
        known_blockers=[
            "Requires Detectron2 environment.",
            "No pip-installable inference path.",
        ],
        recommended_action="expert_sidecar",
    ),
    # =====================================================================
    # SAM family — official HF paths
    # =====================================================================
    "sam-vit-base": ModelSource(
        model_id="sam-vit-base",
        family="sam",
        task="foundation_segment",
        official_repo="https://github.com/facebookresearch/segment-anything",
        hf_repo="facebook/sam-vit-base",
        checkpoint_source="hf_hub",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="SamModel",
        runnable_in_visionservex=True,
        access_status="open",
        recommended_action="add_now",
    ),
    "sam2-hiera-tiny": ModelSource(
        model_id="sam2-hiera-tiny",
        family="sam2",
        task="foundation_segment",
        official_repo="https://github.com/facebookresearch/sam2",
        hf_repo="facebook/sam2-hiera-tiny",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="Sam2Model",
        runnable_in_visionservex=True,
        access_status="open",
        recommended_action="add_now",
    ),
    "sam3-base": ModelSource(
        model_id="sam3-base",
        family="sam3",
        task="foundation_segment",
        official_repo="https://github.com/facebookresearch/sam3",
        official_docs="https://ai.meta.com/blog/segment-anything-model-3/",
        hf_repo="facebook/sam3",
        license="Apache-2.0",
        license_risk="check",
        runnable_in_visionservex=False,
        access_status="gated",
        known_blockers=[
            "Access may be gated at the HF namespace.",
            "VisionServeX does not auto-pull gated weights.",
            "Auth-aware wrapper required.",
        ],
        recommended_action="external_api",
    ),
    # =====================================================================
    # DINOv2 — feature backbone (HF, runnable)
    # =====================================================================
    "dinov2-small": ModelSource(
        model_id="dinov2-small",
        family="dinov2",
        task="embed",
        official_repo="https://github.com/facebookresearch/dinov2",
        official_docs="https://huggingface.co/docs/transformers/en/model_doc/dinov2",
        hf_repo="facebook/dinov2-small",
        checkpoint_source="hf_hub",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="AutoModel",
        runnable_in_visionservex=True,
        access_status="open",
        domain="feature_backbone",
        recommended_action="add_now",
        notes="384-d embedding (ViT-S/14). Use for retrieval, dedup, dataset reports.",
    ),
    "dinov2-base": ModelSource(
        model_id="dinov2-base",
        family="dinov2",
        task="embed",
        official_repo="https://github.com/facebookresearch/dinov2",
        official_docs="https://huggingface.co/docs/transformers/en/model_doc/dinov2",
        hf_repo="facebook/dinov2-base",
        checkpoint_source="hf_hub",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="AutoModel",
        runnable_in_visionservex=True,
        access_status="open",
        domain="feature_backbone",
        recommended_action="add_now",
        notes="768-d embedding (ViT-B/14). Production retrieval default.",
    ),
    "dinov2-large": ModelSource(
        model_id="dinov2-large",
        family="dinov2",
        task="embed",
        official_repo="https://github.com/facebookresearch/dinov2",
        hf_repo="facebook/dinov2-large",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="AutoModel",
        runnable_in_visionservex=True,
        access_status="open",
        domain="feature_backbone",
        recommended_action="add_now",
        notes="1024-d embedding (ViT-L/14). GPU recommended.",
    ),
    "dinov2-giant": ModelSource(
        model_id="dinov2-giant",
        family="dinov2",
        task="embed",
        official_repo="https://github.com/facebookresearch/dinov2",
        hf_repo="facebook/dinov2-giant",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="AutoModel",
        runnable_in_visionservex=True,
        access_status="open",
        domain="feature_backbone",
        recommended_action="add_now",
        notes="1536-d embedding (ViT-g/14). Heavy — use --isolate-process.",
    ),
    "dinov3-vitb16": ModelSource(
        model_id="dinov3-vitb16",
        family="dinov3",
        task="embed",
        official_repo="https://github.com/facebookresearch/dinov3",
        hf_repo="facebook/dinov3-vitb16-pretrain-lvd1689m",
        checkpoint_trust_level="unverified",
        license="Apache-2.0",
        license_risk="check",
        runnable_in_visionservex=False,
        access_status="open",
        domain="feature_backbone",
        known_blockers=[
            "DINOv3 HF model card name not verified live.",
            "Use DINOv2 until DINOv3 access is confirmed.",
        ],
        recommended_action="audit_only",
    ),
    # =====================================================================
    # Florence-2 — VLM (HF, RUNNABLE via Florence2Engine in v1.8.0)
    # =====================================================================
    "florence-2-base": ModelSource(
        model_id="florence-2-base",
        family="florence-2",
        task="vlm",
        official_repo="https://github.com/microsoft/Florence-2",
        hf_repo="microsoft/Florence-2-base",
        checkpoint_source="hf_hub",
        checkpoint_trust_level="official_hf",
        license="MIT",
        install_command="pip install 'visionservex[hf]'",
        hf_class="AutoModelForCausalLM",
        runnable_in_visionservex=True,
        access_status="open",
        domain="open_vocab",
        recommended_action="add_now",
        notes=(
            "Wired in v1.8.0 via Florence2Engine. Task tokens: <CAPTION>, "
            "<DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>, <OD>, "
            "<DENSE_REGION_CAPTION>, <CAPTION_TO_PHRASE_GROUNDING>, <OCR>, "
            "<OCR_WITH_REGION>. trust_remote_code is enabled automatically."
        ),
    ),
    "florence-2-large": ModelSource(
        model_id="florence-2-large",
        family="florence-2",
        task="vlm",
        official_repo="https://github.com/microsoft/Florence-2",
        hf_repo="microsoft/Florence-2-large",
        checkpoint_trust_level="official_hf",
        license="MIT",
        install_command="pip install 'visionservex[hf]'",
        hf_class="AutoModelForCausalLM",
        runnable_in_visionservex=True,
        access_status="open",
        domain="open_vocab",
        recommended_action="add_now",
        notes="Same engine as florence-2-base. Heavier weights; recommend GPU.",
    ),
    # =====================================================================
    # OWLv2 — open-vocabulary detection (HF, RUNNABLE via OWLv2Engine in v1.8.0)
    # =====================================================================
    "owlv2-base-patch16": ModelSource(
        model_id="owlv2-base-patch16",
        family="owlv2",
        task="open_vocab_detect",
        official_docs="https://huggingface.co/docs/transformers/en/model_doc/owlv2",
        hf_repo="google/owlv2-base-patch16-ensemble",
        checkpoint_source="hf_hub",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="Owlv2ForObjectDetection",
        runnable_in_visionservex=True,
        access_status="open",
        domain="open_vocab",
        recommended_action="add_now",
        notes=(
            "Wired in v1.8.0 via OWLv2Engine. Pass prompts as a list or "
            "comma-separated string. Returns OpenVocabularyResult."
        ),
    ),
    "owlv2-large-patch14": ModelSource(
        model_id="owlv2-large-patch14",
        family="owlv2",
        task="open_vocab_detect",
        official_docs="https://huggingface.co/docs/transformers/en/model_doc/owlv2",
        hf_repo="google/owlv2-large-patch14-ensemble",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="Owlv2ForObjectDetection",
        runnable_in_visionservex=True,
        access_status="open",
        domain="open_vocab",
        recommended_action="add_now",
        notes="Same engine as owlv2-base-patch16. Heavier; recommend GPU.",
    ),
    # =====================================================================
    # SigLIP2 — text-image retrieval (HF, runnable)
    # =====================================================================
    "siglip2-base-patch16-224": ModelSource(
        model_id="siglip2-base-patch16-224",
        family="siglip2",
        task="embed",
        official_docs="https://huggingface.co/docs/transformers/en/model_doc/siglip2",
        hf_repo="google/siglip2-base-patch16-224",
        checkpoint_source="hf_hub",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="AutoModel",
        runnable_in_visionservex=True,
        access_status="open",
        domain="feature_backbone",
        recommended_action="add_now",
        notes="Text+image retrieval. Use for surveillance text search.",
    ),
    # =====================================================================
    # Grounding DINO external API
    # =====================================================================
    "grounding-dino-1.5-pro": ModelSource(
        model_id="grounding-dino-1.5-pro",
        family="grounding-dino",
        task="open_vocab_detect",
        official_repo="https://github.com/IDEA-Research/Grounding-DINO-1.5-API",
        license="Custom",
        license_risk="api_only",
        runnable_in_visionservex=False,
        access_status="api_token",
        domain="open_vocab",
        known_blockers=["API-gated. Requires IDEA-Research token."],
        recommended_action="external_api",
    ),
    "grounding-dino-1.6-pro": ModelSource(
        model_id="grounding-dino-1.6-pro",
        family="grounding-dino",
        task="open_vocab_detect",
        official_repo="https://github.com/IDEA-Research/Grounding-DINO-1.5-API",
        license="Custom",
        license_risk="api_only",
        runnable_in_visionservex=False,
        access_status="api_token",
        domain="open_vocab",
        known_blockers=["API-gated. Requires IDEA-Research token."],
        recommended_action="external_api",
    ),
    "dino-x-api": ModelSource(
        model_id="dino-x-api",
        family="dino-x",
        task="open_vocab_detect",
        official_repo="https://github.com/IDEA-Research/Grounding-DINO-1.5-API",
        license="Custom",
        license_risk="api_only",
        runnable_in_visionservex=False,
        access_status="api_token",
        recommended_action="external_api",
    ),
    # =====================================================================
    # Anomalib (industrial) — expert sidecar
    # =====================================================================
    "anomalib-patchcore": ModelSource(
        model_id="anomalib-patchcore",
        family="anomalib",
        task="anomaly",
        official_repo="https://github.com/open-edge-platform/anomalib",
        official_docs="https://anomalib.readthedocs.io/",
        license="Apache-2.0",
        install_command="pip install anomalib  # not in visionservex core",
        runnable_in_visionservex=False,
        access_status="open",
        domain="industrial",
        known_blockers=[
            "anomalib is heavy; kept outside core dependencies.",
            "Pipeline integration not yet wired.",
        ],
        recommended_action="expert_sidecar",
    ),
    # =====================================================================
    # Person Re-ID — expert sidecar (surveillance)
    # =====================================================================
    "osnet-x1.0": ModelSource(
        model_id="osnet-x1.0",
        family="osnet",
        task="reid",
        official_repo="https://github.com/KaiyangZhou/deep-person-reid",
        license="MIT",
        install_command="pip install torchreid  # not in core",
        runnable_in_visionservex=False,
        access_status="open",
        domain="surveillance",
        known_blockers=["torchreid not in core dependencies."],
        recommended_action="expert_sidecar",
    ),
    "bytetrack": ModelSource(
        model_id="bytetrack",
        family="bytetrack",
        task="track",
        official_repo="https://github.com/ifzhang/ByteTrack",
        license="MIT",
        license_risk="check",
        runnable_in_visionservex=False,
        access_status="open",
        domain="surveillance",
        known_blockers=["ByteTrack not packaged; tracker pipeline pending."],
        recommended_action="expert_sidecar",
    ),
    # =====================================================================
    # Medical (license care!)
    # =====================================================================
    "totalsegmentator": ModelSource(
        model_id="totalsegmentator",
        family="totalsegmentator",
        task="segment",
        official_repo="https://github.com/wasserth/TotalSegmentator",
        license="Apache-2.0",
        license_risk="check",  # certain submodels may be restricted
        install_command="pip install totalsegmentator  # not in core",
        runnable_in_visionservex=False,
        access_status="open",
        domain="medical",
        known_blockers=[
            "Heavy dependency on nibabel, SimpleITK, etc.",
            "License has commercial restrictions on some submodels.",
        ],
        recommended_action="non_core_license_optional",
    ),
    "medsam": ModelSource(
        model_id="medsam",
        family="medsam",
        task="foundation_segment",
        official_repo="https://github.com/bowang-lab/MedSAM",
        hf_repo="wanglab/medsam-vit-base",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        license_risk="check",
        install_command="pip install 'visionservex[hf]'",
        hf_class="SamModel",
        runnable_in_visionservex=True,
        access_status="open",
        domain="medical",
        recommended_action="add_now",
        notes=(
            "Wired in v2.1.0 via sam_hf engine (standard SamModel). "
            "Requires box/point prompt. "
            "RESEARCH AND EDUCATION ONLY — no diagnostic claims."
        ),
    ),
    "medsam2": ModelSource(
        model_id="medsam2",
        family="medsam2",
        task="foundation_segment",
        official_repo="https://github.com/bowang-lab/MedSAM2",
        license="Apache-2.0",
        license_risk="check",
        runnable_in_visionservex=False,
        access_status="open",
        domain="medical",
        recommended_action="expert_sidecar",
    ),
    "nnunet-v2": ModelSource(
        model_id="nnunet-v2",
        family="nnunet",
        task="segment",
        official_repo="https://github.com/MIC-DKFZ/nnUNet",
        license="Apache-2.0",
        runnable_in_visionservex=False,
        access_status="open",
        domain="medical",
        known_blockers=["Training/inference framework, not a single model."],
        recommended_action="expert_sidecar",
    ),
    # =====================================================================
    # Aerial / geospatial
    # =====================================================================
    "rtmdet-r2-s": ModelSource(
        model_id="rtmdet-r2-s",
        family="rtmdet",
        task="obb",
        official_repo="https://github.com/open-mmlab/mmrotate",
        license="Apache-2.0",
        runnable_in_visionservex=False,
        access_status="open",
        domain="aerial",
        known_blockers=["Requires OpenMMLab MMRotate sidecar."],
        recommended_action="expert_sidecar",
    ),
    "prithvi-eo-2.0": ModelSource(
        model_id="prithvi-eo-2.0",
        family="prithvi",
        task="embed",
        official_repo="https://github.com/NASA-IMPACT/Prithvi-EO-2.0",
        hf_repo="ibm-nasa-geospatial",
        license="Apache-2.0",
        license_risk="check",
        runnable_in_visionservex=False,
        access_status="open",
        domain="aerial",
        known_blockers=[
            "Multispectral input — not standard RGB.",
            "Geospatial extra; not in core.",
        ],
        recommended_action="non_core_license_optional",
    ),
    # =====================================================================
    # Agriculture
    # =====================================================================
    "agriclip": ModelSource(
        model_id="agriclip",
        family="agriclip",
        task="embed",
        official_repo="https://github.com/umair1221/AgriCLIP",
        license="check",
        license_risk="check",
        runnable_in_visionservex=False,
        access_status="open",
        domain="agriculture",
        known_blockers=["License and HF availability not verified live."],
        recommended_action="audit_only",
    ),
    # =====================================================================
    # YOLO-World — license issue (likely GPL/AGPL)
    # =====================================================================
    "yolo-world": ModelSource(
        model_id="yolo-world",
        family="yolo-world",
        task="open_vocab_detect",
        official_repo="https://github.com/AILab-CVC/YOLO-World",
        license="check",
        license_risk="restricted",
        runnable_in_visionservex=False,
        access_status="open",
        domain="open_vocab",
        known_blockers=[
            "License likely GPL/AGPL — excluded from permissive Apache/MIT core.",
            "Use Grounding DINO or OWLv2 instead.",
        ],
        recommended_action="do_not_add",
    ),
    # =====================================================================
    # OWL-ViT v1 — RUNNABLE in v2.1.0 via OWLv2Engine
    # =====================================================================
    "owlvit-base-patch32": ModelSource(
        model_id="owlvit-base-patch32",
        family="owlvit",
        task="open_vocab_detect",
        official_docs="https://huggingface.co/docs/transformers/en/model_doc/owlvit",
        hf_repo="google/owlvit-base-patch32",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="OwlViTForObjectDetection",
        runnable_in_visionservex=True,
        access_status="open",
        domain="open_vocab",
        recommended_action="add_now",
        notes="Wired in v2.1.0 via OWLv2Engine (owlvit alias). OWL-ViT v1 predecessor to OWLv2.",
    ),
    "owlvit-large-patch14": ModelSource(
        model_id="owlvit-large-patch14",
        family="owlvit",
        task="open_vocab_detect",
        official_docs="https://huggingface.co/docs/transformers/en/model_doc/owlvit",
        hf_repo="google/owlvit-large-patch14",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="OwlViTForObjectDetection",
        runnable_in_visionservex=True,
        access_status="open",
        domain="open_vocab",
        recommended_action="add_now",
        notes="OWL-ViT v1 large. Same engine as owlvit-base-patch32.",
    ),
    # =====================================================================
    # CLIP — RUNNABLE in v2.1.0 via DINOv2Engine (clip alias)
    # =====================================================================
    "clip-vit-base-patch32": ModelSource(
        model_id="clip-vit-base-patch32",
        family="clip",
        task="embed",
        official_repo="https://github.com/openai/CLIP",
        hf_repo="openai/clip-vit-base-patch32",
        checkpoint_trust_level="official_hf",
        license="MIT",
        install_command="pip install 'visionservex[hf]'",
        hf_class="CLIPModel",
        runnable_in_visionservex=True,
        access_status="open",
        domain="feature_backbone",
        recommended_action="add_now",
        notes="Wired in v2.1.0 via DINOv2Engine (clip alias). Image-side embedding via vision_model.",
    ),
    "clip-vit-large-patch14": ModelSource(
        model_id="clip-vit-large-patch14",
        family="clip",
        task="embed",
        official_repo="https://github.com/openai/CLIP",
        hf_repo="openai/clip-vit-large-patch14",
        checkpoint_trust_level="official_hf",
        license="MIT",
        install_command="pip install 'visionservex[hf]'",
        hf_class="CLIPModel",
        runnable_in_visionservex=True,
        access_status="open",
        domain="feature_backbone",
        recommended_action="add_now",
        notes="CLIP ViT-L/14. Higher quality embeddings.",
    ),
    # =====================================================================
    # SigLIP v1 — RUNNABLE via DINOv2Engine (siglip alias)
    # =====================================================================
    "siglip-base-patch16-224": ModelSource(
        model_id="siglip-base-patch16-224",
        family="siglip",
        task="embed",
        official_docs="https://huggingface.co/docs/transformers/en/model_doc/siglip",
        hf_repo="google/siglip-base-patch16-224",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="SiglipModel",
        runnable_in_visionservex=True,
        access_status="open",
        domain="feature_backbone",
        recommended_action="add_now",
        notes="SigLIP v1 base. Wired via DINOv2Engine (siglip alias).",
    ),
    # =====================================================================
    # ConvNeXtV2 — RUNNABLE via HFClassifyEngine
    # =====================================================================
    "convnextv2-tiny": ModelSource(
        model_id="convnextv2-tiny",
        family="convnextv2",
        task="classify",
        official_repo="https://github.com/facebookresearch/ConvNeXt-V2",
        hf_repo="facebook/convnextv2-tiny-22k-224",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="AutoModelForImageClassification",
        runnable_in_visionservex=True,
        access_status="open",
        domain="general",
        recommended_action="add_now",
        notes="Wired in v2.1.0 via HFClassifyEngine. ImageNet-22k pretrained, top-k classification.",
    ),
    "convnextv2-base": ModelSource(
        model_id="convnextv2-base",
        family="convnextv2",
        task="classify",
        official_repo="https://github.com/facebookresearch/ConvNeXt-V2",
        hf_repo="facebook/convnextv2-base-22k-224",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="AutoModelForImageClassification",
        runnable_in_visionservex=True,
        access_status="open",
        domain="general",
        recommended_action="add_now",
        notes="ConvNeXtV2-Base. Good accuracy/latency balance.",
    ),
    "convnextv2-large": ModelSource(
        model_id="convnextv2-large",
        family="convnextv2",
        task="classify",
        official_repo="https://github.com/facebookresearch/ConvNeXt-V2",
        hf_repo="facebook/convnextv2-large-22k-224",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="AutoModelForImageClassification",
        runnable_in_visionservex=True,
        access_status="open",
        domain="general",
        recommended_action="add_now",
        notes="ConvNeXtV2-Large.",
    ),
    # =====================================================================
    # SAM2.1 — RUNNABLE in v2.2.0 via sam2_hf engine (HF IDs verified)
    # =====================================================================
    "sam2.1-hiera-tiny": ModelSource(
        model_id="sam2.1-hiera-tiny",
        family="sam2.1",
        task="foundation_segment",
        official_repo="https://github.com/facebookresearch/sam2",
        hf_repo="facebook/sam2.1-hiera-tiny",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="Sam2Model",
        runnable_in_visionservex=True,
        access_status="open",
        domain="promptable_foundation",
        recommended_action="add_now",
        notes=(
            "SAM 2.1 Hiera-Tiny. HF model ID confirmed: facebook/sam2.1-hiera-tiny (Apache-2.0). "
            "Wired in v2.2.0 via sam2_hf engine. Improved object permanence vs SAM 2."
        ),
    ),
    "sam2.1-hiera-small": ModelSource(
        model_id="sam2.1-hiera-small",
        family="sam2.1",
        task="foundation_segment",
        official_repo="https://github.com/facebookresearch/sam2",
        hf_repo="facebook/sam2.1-hiera-small",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="Sam2Model",
        runnable_in_visionservex=True,
        access_status="open",
        domain="promptable_foundation",
        recommended_action="add_now",
    ),
    "sam2.1-hiera-base-plus": ModelSource(
        model_id="sam2.1-hiera-base-plus",
        family="sam2.1",
        task="foundation_segment",
        official_repo="https://github.com/facebookresearch/sam2",
        hf_repo="facebook/sam2.1-hiera-base-plus",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="Sam2Model",
        runnable_in_visionservex=True,
        access_status="open",
        domain="promptable_foundation",
        recommended_action="add_now",
    ),
    "sam2.1-hiera-large": ModelSource(
        model_id="sam2.1-hiera-large",
        family="sam2.1",
        task="foundation_segment",
        official_repo="https://github.com/facebookresearch/sam2",
        hf_repo="facebook/sam2.1-hiera-large",
        checkpoint_trust_level="official_hf",
        license="Apache-2.0",
        install_command="pip install 'visionservex[hf]'",
        hf_class="Sam2Model",
        runnable_in_visionservex=True,
        access_status="open",
        domain="promptable_foundation",
        recommended_action="add_now",
        notes="SAM 2.1 large. Highest quality; GPU required.",
    ),
    # =====================================================================
    # Lightweight SAM alternatives — license-audited decisions (v2.2.0)
    # =====================================================================
    "fastsam-s": ModelSource(
        model_id="fastsam-s",
        family="fastsam",
        task="foundation_segment",
        official_repo="https://github.com/CASIA-IVA-Lab/FastSAM",
        license="AGPL-3.0",
        license_risk="agpl",
        install_command="git clone https://github.com/CASIA-IVA-Lab/FastSAM.git",
        runnable_in_visionservex=False,
        access_status="open",
        domain="general",
        known_blockers=[
            "License is AGPL-3.0. VisionServeX core requires Apache-2.0/MIT-compatible licenses.",
            "AGPL-3.0 requires downstream users to open-source their applications if served over a network.",
        ],
        recommended_action="do_not_add",
        notes="FastSAM is AGPL-3.0. Excluded from VisionServeX core.",
    ),
    "fastsam-x": ModelSource(
        model_id="fastsam-x",
        family="fastsam",
        task="foundation_segment",
        official_repo="https://github.com/CASIA-IVA-Lab/FastSAM",
        license="AGPL-3.0",
        license_risk="agpl",
        install_command="git clone https://github.com/CASIA-IVA-Lab/FastSAM.git",
        runnable_in_visionservex=False,
        access_status="open",
        domain="general",
        known_blockers=["License AGPL-3.0 — excluded from core."],
        recommended_action="do_not_add",
    ),
    "mobilesam": ModelSource(
        model_id="mobilesam",
        family="mobilesam",
        task="foundation_segment",
        official_repo="https://github.com/ChaoningZhang/MobileSAM",
        license="Apache-2.0",
        license_risk="none",
        install_command="pip install mobile-sam  # or: git clone https://github.com/ChaoningZhang/MobileSAM",
        runnable_in_visionservex=False,
        access_status="open",
        domain="general",
        known_blockers=[
            "No HF Hub model card for pip-installable weights. Checkpoint distributed via GitHub release.",
            "Not yet wired in VisionServeX engine; manual download + sam_hf adapter path needed.",
        ],
        recommended_action="expert_sidecar",
        notes="MobileSAM is Apache-2.0. Install: pip install mobile-sam. No HF Hub model; checkpoint at github.com/ChaoningZhang/MobileSAM/releases.",
    ),
    "efficientsam": ModelSource(
        model_id="efficientsam",
        family="efficientsam",
        task="foundation_segment",
        official_repo="https://github.com/yformer/EfficientSAM",
        license="Apache-2.0",
        license_risk="none",
        install_command="git clone https://github.com/yformer/EfficientSAM.git && pip install -e .",
        runnable_in_visionservex=False,
        access_status="open",
        domain="general",
        known_blockers=[
            "No HF Hub model; GitHub-only install.",
            "Not yet wired in VisionServeX engine.",
        ],
        recommended_action="expert_sidecar",
        notes="EfficientSAM is Apache-2.0. Requires GitHub install.",
    ),
    "hq-sam": ModelSource(
        model_id="hq-sam",
        family="hq-sam",
        task="foundation_segment",
        official_repo="https://github.com/SysCV/sam-hq",
        license="Apache-2.0",
        license_risk="none",
        install_command="pip install segment-anything-hq  # check latest release",
        runnable_in_visionservex=False,
        access_status="open",
        domain="general",
        known_blockers=[
            "No HF Hub model with standard SAM API compatibility confirmed.",
            "Not yet wired in VisionServeX engine.",
        ],
        recommended_action="expert_sidecar",
        notes="HQ-SAM is Apache-2.0. High-quality mask prediction.",
    ),
    "edgesam": ModelSource(
        model_id="edgesam",
        family="edgesam",
        task="foundation_segment",
        official_repo="https://github.com/chongzhou96/EdgeSAM",
        license="Apache-2.0",
        license_risk="none",
        install_command="git clone https://github.com/chongzhou96/EdgeSAM.git && pip install -e .",
        runnable_in_visionservex=False,
        access_status="open",
        domain="general",
        known_blockers=[
            "GitHub-only install; no HF Hub model.",
            "Targets edge/mobile devices; not prioritized for server deployment.",
        ],
        recommended_action="expert_sidecar",
        notes="EdgeSAM is Apache-2.0. Optimized for edge devices.",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_model_source(model_id: str) -> ModelSource | None:
    """Lookup a model's source manifest entry. Returns None if not in manifest."""
    return SOURCE_MANIFEST.get(model_id)


def list_all_models() -> list[str]:
    """Return all model IDs in the source manifest."""
    return sorted(SOURCE_MANIFEST.keys())


def verify_manifest() -> dict[str, Any]:
    """Static verification of manifest entries (no network calls).

    Returns counts and structural issues. To verify URLs live, use
    `visionservex model-zoo verify-links`.
    """
    issues: list[dict[str, str]] = []
    counts = {
        "total": 0,
        "runnable": 0,
        "external_api": 0,
        "expert_sidecar": 0,
        "audit_only": 0,
        "non_core_license_optional": 0,
        "do_not_add": 0,
        "missing_official_repo": 0,
    }

    for mid, src in SOURCE_MANIFEST.items():
        counts["total"] += 1
        if src.runnable_in_visionservex:
            counts["runnable"] += 1
        counts[src.recommended_action] = counts.get(src.recommended_action, 0) + 1
        if not src.official_repo and src.recommended_action not in ("external_api",):
            counts["missing_official_repo"] += 1
            issues.append({"model_id": mid, "issue": "missing official_repo URL"})

    return {"counts": counts, "issues": issues}


__all__ = [
    "SOURCE_MANIFEST",
    "ModelSource",
    "get_model_source",
    "list_all_models",
    "verify_manifest",
]
