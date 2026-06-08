# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Canonical model license policy for VisionServeX v3.8.

The :data:`POLICY` table is the single source of truth for every model the
project advertises. Each :class:`ModelPolicy` row maps a model to one of nine
``final_policy`` buckets and records the separable facts a downstream user needs
to act lawfully:

* code license vs. weights license vs. dataset/pretraining risk (tracked apart);
* whether the repo is gated and a local Hugging Face token is required;
* whether the *user* must accept an upstream license themselves;
* whether VisionServeX may auto-download the weights and whether it may ship
  them (it never ships gated/restricted weights — and never bundles ANY weights
  into the wheel);
* whether the model is default-safe, commercial-safe, and production-allowed;
* the exact warning text and the exact next command.

Hard invariants (enforced by tests in ``tests/test_v38_commercial_safe_core_policy.py``
and friends):

* A Hugging Face token never implies redistribution rights.
* Gated weights are never packaged into PyPI / GitHub / Docker (``can_ship_weights``
  is False for every gated row — in fact for every row).
* Non-commercial models are never ``production_allowed`` and never ``default_safe``.
* AGPL / enterprise models never enter ``default_safe`` core.
* API-only models are never counted as local models (``is_local`` is False).
* ``legal_review_required`` models are never ``commercial_safe`` until resolved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Final policy vocabulary (the nine buckets).
# --------------------------------------------------------------------------- #
FINAL_POLICIES: tuple[str, ...] = (
    "commercial_safe_core",
    "byot_license_required",
    "auth_required_license_pending",
    "external_api_only_terms_required",
    "noncommercial_restricted",
    "enterprise_license_required",
    "legal_review_required",
    "excluded_from_core",
    "not_released_or_unverifiable",
)

# --------------------------------------------------------------------------- #
# Mandatory warning texts (verbatim — referenced by tests; do not edit wording).
# --------------------------------------------------------------------------- #
WARNING_TEXTS: dict[str, str] = {
    # 1. BYOT gated / custom upstream license
    "byot": (
        "This model is gated or uses a custom upstream license. You must use your "
        "own Hugging Face token and accept the upstream license yourself. "
        "VisionServeX does not redistribute the weights. Commercial use depends on "
        "the upstream license terms you accepted."
    ),
    # 2. Non-commercial / restricted
    "noncommercial": (
        "WARNING: This model is non-commercial/restricted. Do not use it for paid "
        "SaaS, client work, production annotation, or commercial products unless you "
        "have written permission from the model owner."
    ),
    # 3. Enterprise / AGPL / copyleft
    "enterprise": (
        "WARNING: This model requires an enterprise/commercial license or has "
        "AGPL/copyleft obligations. It is disabled in VisionServeX commercial-safe "
        "core."
    ),
    # 4. External API only
    "api": (
        "External API model. Your data may leave the local environment. You must "
        "provide your own provider API key and comply with provider terms."
    ),
    # 5. Legal review
    "legal_review": ("License/provenance is unclear. Legal review required before commercial use."),
    # Benign — commercial-safe core (no warning, informational only).
    "commercial_safe": (
        "Commercial-safe core model (permissive license). Weights are pulled from "
        "the official source on demand; VisionServeX does not bundle them."
    ),
    # Not released / unverifiable
    "not_released": (
        "This model is not released, not found at an official source, or its "
        "provenance could not be verified. It cannot be run or shipped."
    ),
}

# Which warning key each final_policy uses by default.
_POLICY_WARNING = {
    "commercial_safe_core": "commercial_safe",
    "byot_license_required": "byot",
    "auth_required_license_pending": "byot",
    "external_api_only_terms_required": "api",
    "noncommercial_restricted": "noncommercial",
    "enterprise_license_required": "enterprise",
    "legal_review_required": "legal_review",
    "excluded_from_core": "noncommercial",
    "not_released_or_unverifiable": "not_released",
}


@dataclass(frozen=True)
class ModelPolicy:
    """Per-model license policy row (the matrix schema)."""

    model_id: str
    family: str
    final_policy: str
    code_license: str = ""
    weights_license: str = ""
    dataset_risk: str = "low"
    gated: bool = False
    local_token_required: bool = False
    user_license_required: bool = False
    can_auto_download: bool = False
    can_ship_weights: bool = False  # VisionServeX NEVER ships weights -> always False
    default_safe: bool = False
    commercial_safe: bool = False
    production_allowed: bool = False
    is_local: bool = True  # False for external-API-only models
    hf_repo: str = ""
    upstream_url: str = ""
    exact_next_command: str = ""
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def warning_text(self) -> str:
        return WARNING_TEXTS[_POLICY_WARNING[self.final_policy]]

    def as_row(self) -> dict[str, object]:
        """Flat dict for the CSV matrix (stable column order via MATRIX_COLUMNS)."""
        return {
            "model_id": self.model_id,
            "family": self.family,
            "code_license": self.code_license,
            "weights_license": self.weights_license,
            "dataset_risk": self.dataset_risk,
            "gated": self.gated,
            "local_token_required": self.local_token_required,
            "user_license_required": self.user_license_required,
            "can_auto_download": self.can_auto_download,
            "can_ship_weights": self.can_ship_weights,
            "default_safe": self.default_safe,
            "commercial_safe": self.commercial_safe,
            "production_allowed": self.production_allowed,
            "is_local": self.is_local,
            "final_policy": self.final_policy,
            "hf_repo": self.hf_repo,
            "warning_text": self.warning_text,
            "exact_next_command": self.exact_next_command,
            "notes": self.notes,
        }


MATRIX_COLUMNS: tuple[str, ...] = (
    "model_id",
    "family",
    "code_license",
    "weights_license",
    "dataset_risk",
    "gated",
    "local_token_required",
    "user_license_required",
    "can_auto_download",
    "can_ship_weights",
    "default_safe",
    "commercial_safe",
    "production_allowed",
    "is_local",
    "final_policy",
    "hf_repo",
    "warning_text",
    "exact_next_command",
    "notes",
)


# --------------------------------------------------------------------------- #
# Builders — keep defaults consistent per bucket so invariants can't be missed.
# --------------------------------------------------------------------------- #
def _core(
    model_id,
    family,
    *,
    code="Apache-2.0",
    weights="Apache-2.0",
    dataset_risk="low",
    hf_repo="",
    upstream="",
    nxt="",
    notes="",
    aliases=(),
):
    """commercial_safe_core: permissive, default-safe, production-allowed.

    Weights are pulled from the official source on demand. VisionServeX still
    never *bundles* weights, so ``can_ship_weights`` stays False.
    """
    return ModelPolicy(
        model_id=model_id,
        family=family,
        final_policy="commercial_safe_core",
        code_license=code,
        weights_license=weights,
        dataset_risk=dataset_risk,
        gated=False,
        local_token_required=False,
        user_license_required=False,
        can_auto_download=True,
        can_ship_weights=False,
        default_safe=True,
        commercial_safe=True,
        production_allowed=True,
        is_local=True,
        hf_repo=hf_repo,
        upstream_url=upstream,
        exact_next_command=nxt or f"visionservex model pull {model_id}",
        notes=notes,
        aliases=tuple(aliases),
    )


def _byot(
    model_id,
    family,
    *,
    code,
    weights,
    dataset_risk="medium",
    hf_repo,
    upstream="",
    nxt="",
    notes="",
    aliases=(),
):
    """byot_license_required: gated, user supplies token + accepts upstream license."""
    return ModelPolicy(
        model_id=model_id,
        family=family,
        final_policy="byot_license_required",
        code_license=code,
        weights_license=weights,
        dataset_risk=dataset_risk,
        gated=True,
        local_token_required=True,
        user_license_required=True,
        can_auto_download=False,
        can_ship_weights=False,
        default_safe=False,
        commercial_safe=False,
        production_allowed=False,
        is_local=True,
        hf_repo=hf_repo,
        upstream_url=upstream,
        exact_next_command=(
            nxt
            or f"visionservex hf connect && visionservex model pull {model_id} "
            f"--accept-upstream-license"
        ),
        notes=notes,
        aliases=tuple(aliases),
    )


def _api(
    model_id,
    family,
    *,
    code="proprietary (hosted API)",
    weights="not released",
    provider_env="DINOX_API_KEY",
    upstream="",
    nxt="",
    notes="",
    aliases=(),
):
    """external_api_only_terms_required: no local weights; data leaves the box."""
    return ModelPolicy(
        model_id=model_id,
        family=family,
        final_policy="external_api_only_terms_required",
        code_license=code,
        weights_license=weights,
        dataset_risk="n/a",
        gated=False,
        local_token_required=False,
        user_license_required=True,
        can_auto_download=False,
        can_ship_weights=False,
        default_safe=False,
        commercial_safe=False,
        production_allowed=False,
        is_local=False,
        hf_repo="",
        upstream_url=upstream,
        exact_next_command=nxt
        or f"export {provider_env}=... && visionservex model license {model_id}",
        notes=notes,
        aliases=tuple(aliases),
    )


def _noncommercial(
    model_id,
    family,
    *,
    code,
    weights,
    dataset_risk="high",
    gated=False,
    hf_repo="",
    upstream="",
    nxt="",
    notes="",
    aliases=(),
):
    """noncommercial_restricted: never production / default-safe / commercial."""
    return ModelPolicy(
        model_id=model_id,
        family=family,
        final_policy="noncommercial_restricted",
        code_license=code,
        weights_license=weights,
        dataset_risk=dataset_risk,
        gated=gated,
        local_token_required=gated,
        user_license_required=True,
        can_auto_download=False,
        can_ship_weights=False,
        default_safe=False,
        commercial_safe=False,
        production_allowed=False,
        is_local=True,
        hf_repo=hf_repo,
        upstream_url=upstream,
        exact_next_command=nxt
        or f"visionservex model license {model_id}  # non-commercial; research-only",
        notes=notes,
        aliases=tuple(aliases),
    )


def _enterprise(model_id, family, *, code, weights, upstream="", nxt="", notes="", aliases=()):
    """enterprise_license_required: AGPL / enterprise-only; disabled in core."""
    return ModelPolicy(
        model_id=model_id,
        family=family,
        final_policy="enterprise_license_required",
        code_license=code,
        weights_license=weights,
        dataset_risk="medium",
        gated=False,
        local_token_required=False,
        user_license_required=True,
        can_auto_download=False,
        can_ship_weights=False,
        default_safe=False,
        commercial_safe=False,
        production_allowed=False,
        is_local=True,
        hf_repo="",
        upstream_url=upstream,
        exact_next_command=nxt
        or f"visionservex model license {model_id}  # enterprise/AGPL license required",
        notes=notes,
        aliases=tuple(aliases),
    )


def _legal(
    model_id,
    family,
    *,
    code,
    weights,
    dataset_risk="high",
    gated=False,
    hf_repo="",
    upstream="",
    nxt="",
    notes="",
    aliases=(),
):
    """legal_review_required: provenance unclear; not commercial-safe until resolved."""
    return ModelPolicy(
        model_id=model_id,
        family=family,
        final_policy="legal_review_required",
        code_license=code,
        weights_license=weights,
        dataset_risk=dataset_risk,
        gated=gated,
        local_token_required=gated,
        user_license_required=True,
        can_auto_download=False,
        can_ship_weights=False,
        default_safe=False,
        commercial_safe=False,
        production_allowed=False,
        is_local=True,
        hf_repo=hf_repo,
        upstream_url=upstream,
        exact_next_command=nxt or f"visionservex legal review {model_id}",
        notes=notes,
        aliases=tuple(aliases),
    )


def _not_released(model_id, family, *, notes="", upstream="", aliases=()):
    """not_released_or_unverifiable: cannot run or ship."""
    return ModelPolicy(
        model_id=model_id,
        family=family,
        final_policy="not_released_or_unverifiable",
        code_license="unknown",
        weights_license="unknown",
        dataset_risk="unknown",
        gated=False,
        local_token_required=False,
        user_license_required=False,
        can_auto_download=False,
        can_ship_weights=False,
        default_safe=False,
        commercial_safe=False,
        production_allowed=False,
        is_local=True,
        hf_repo="",
        upstream_url=upstream,
        exact_next_command=f"visionservex model status {model_id} --explain",
        notes=notes,
        aliases=tuple(aliases),
    )


# --------------------------------------------------------------------------- #
# THE POLICY TABLE
# Backbone: v37_license_decisions.csv (adversarial 9-agent research) +
# the v3.8 authoritative known-state license audit.
# --------------------------------------------------------------------------- #
_ROWS: list[ModelPolicy] = []

# ----- commercial_safe_core: permissive, runnable, production-allowed --------
_GH = "https://github.com"
_ROWS += [
    # SAM v1 (Apache-2.0, SA-1B industry-accepted same as Meta's own release)
    _core(
        "sam-vit-base",
        "sam",
        hf_repo="facebook/sam-vit-base",
        upstream=f"{_GH}/facebookresearch/segment-anything",
        dataset_risk="low",
        aliases=("sam-vit-b",),
    ),
    _core("sam-vit-large", "sam", hf_repo="facebook/sam-vit-large", aliases=("sam-vit-l",)),
    _core("sam-vit-huge", "sam", hf_repo="facebook/sam-vit-huge", aliases=("sam-vit-h",)),
    _core(
        "mobilesam",
        "sam",
        upstream=f"{_GH}/ChaoningZhang/MobileSAM",
        notes="MobileSAM Apache-2.0; ONNX-exportable.",
    ),
    _core(
        "efficientsam",
        "sam",
        upstream=f"{_GH}/yformer/EfficientSAM",
        notes="EfficientSAM Apache-2.0; SA-1B distillation (industry-accepted).",
    ),
    # SAM2 / SAM2.1 (Apache-2.0)
    _core("sam2-hiera-tiny", "sam2", hf_repo="facebook/sam2-hiera-tiny"),
    _core("sam2-hiera-small", "sam2", hf_repo="facebook/sam2-hiera-small"),
    _core("sam2-hiera-base-plus", "sam2", hf_repo="facebook/sam2-hiera-base-plus"),
    _core("sam2-hiera-large", "sam2", hf_repo="facebook/sam2-hiera-large"),
    _core("sam2.1-hiera-tiny", "sam2.1", hf_repo="facebook/sam2.1-hiera-tiny"),
    _core("sam2.1-hiera-small", "sam2.1", hf_repo="facebook/sam2.1-hiera-small"),
    _core("sam2.1-hiera-base-plus", "sam2.1", hf_repo="facebook/sam2.1-hiera-base-plus"),
    _core("sam2.1-hiera-large", "sam2.1", hf_repo="facebook/sam2.1-hiera-large"),
    # DINOv2 (Apache-2.0 — distinct from DINOv3 custom license)
    _core("dinov2-small", "dinov2", hf_repo="facebook/dinov2-small"),
    _core("dinov2-base", "dinov2", hf_repo="facebook/dinov2-base"),
    _core("dinov2-large", "dinov2", hf_repo="facebook/dinov2-large"),
    _core("dinov2-giant", "dinov2", hf_repo="facebook/dinov2-giant"),
    # Open GroundingDINO (Apache-2.0 weights — the released checkpoints)
    _core("grounding-dino-tiny", "grounding-dino", hf_repo="IDEA-Research/grounding-dino-tiny"),
    _core("grounding-dino-base", "grounding-dino", hf_repo="IDEA-Research/grounding-dino-base"),
    _core("grounding-dino-swin-t", "grounding-dino", upstream=f"{_GH}/IDEA-Research/GroundingDINO"),
    _core("grounding-dino-swin-b", "grounding-dino", upstream=f"{_GH}/IDEA-Research/GroundingDINO"),
    # Florence-2 (MIT)
    _core(
        "florence-2-base",
        "florence2",
        code="MIT",
        weights="MIT",
        hf_repo="microsoft/Florence-2-base",
    ),
    _core(
        "florence-2-large",
        "florence2",
        code="MIT",
        weights="MIT",
        hf_repo="microsoft/Florence-2-large",
    ),
    # CLIP (MIT/OpenAI), OWL-ViT / OWLv2 (Apache-2.0)
    _core(
        "clip-vit-base-patch32",
        "clip",
        code="MIT",
        weights="MIT",
        hf_repo="openai/clip-vit-base-patch32",
        aliases=("clip",),
    ),
    _core(
        "owlvit-base-patch32", "owlvit", hf_repo="google/owlvit-base-patch32", aliases=("owl-vit",)
    ),
    _core(
        "owlv2-base-patch16-ensemble",
        "owlv2",
        hf_repo="google/owlv2-base-patch16-ensemble",
        aliases=("owlv2",),
    ),
    # DepthAnything small only (Apache-2.0). Larger V2 -> non-commercial below.
    _core(
        "depth-anything-small",
        "depth-anything",
        hf_repo="LiheYoung/depth-anything-small-hf",
        aliases=("depthanything-small", "depth-anything-v2-small"),
    ),
    # RF-DETR-Seg core variants (Apache-2.0; DINOv2 backbone; COCO training)
    _core("rfdetr-seg-nano", "rf-detr", upstream=f"{_GH}/roboflow/rf-detr", dataset_risk="low"),
    _core("rfdetr-seg-small", "rf-detr", upstream=f"{_GH}/roboflow/rf-detr"),
    _core("rfdetr-seg-medium", "rf-detr", upstream=f"{_GH}/roboflow/rf-detr"),
    _core("rfdetr-seg-large", "rf-detr", upstream=f"{_GH}/roboflow/rf-detr"),
    # EfficientViT-SAM (MIT Han Lab, Apache-2.0; SA-1B distillation documented)
    _core(
        "efficientvit-sam-l0",
        "efficient-sam",
        upstream=f"{_GH}/mit-han-lab/efficientvit",
        dataset_risk="medium",
        notes="Apache-2.0 weights; SA-1B research-only dataset provenance documented.",
    ),
    _core(
        "efficientvit-sam-l1",
        "efficient-sam",
        upstream=f"{_GH}/mit-han-lab/efficientvit",
        dataset_risk="medium",
    ),
    _core(
        "efficientvit-sam-l2",
        "efficient-sam",
        upstream=f"{_GH}/mit-han-lab/efficientvit",
        dataset_risk="medium",
    ),
    # RITM interactive segmentation (MIT code + permissive datasets)
    _core(
        "ritm",
        "interactive-seg",
        code="MIT",
        weights="MIT",
        dataset_risk="low",
        upstream=f"{_GH}/SamsungLabs/ritm_interactive_segmentation",
        nxt="visionservex model pull ritm  # MIT; user-supplied checkpoint path ok",
        notes="MIT code, SBD/COCO+LVIS datasets; no NC backbone. Checkpoint user-supplied.",
    ),
    # Detection sidecars — commercial-safe core *if their runtime works* (Apache-2.0).
    _core(
        "maskdino",
        "maskdino",
        upstream=f"{_GH}/IDEA-Research/MaskDINO",
        dataset_risk="low",
        notes="Apache-2.0; commercial-safe core when the Detectron2 sidecar runtime builds.",
    ),
    _core(
        "co-dino",
        "co-detr",
        upstream=f"{_GH}/Sense-X/Co-DETR",
        notes="Apache-2.0; commercial-safe core when the mmdet sidecar runtime builds.",
        aliases=("codino", "co-detr"),
    ),
    _core(
        "rt-detrv4",
        "rt-detr",
        upstream=f"{_GH}/RT-DETRs/RT-DETRv4",
        dataset_risk="low",
        notes="Apache-2.0; checkpoint via official source; core when runtime works.",
        aliases=("rtdetrv4",),
    ),
    _core(
        "rtmdet",
        "rtmdet",
        upstream=f"{_GH}/open-mmlab/mmdetection",
        notes="Apache-2.0 (OpenMMLab); core when the mmdet sidecar runtime builds.",
    ),
]

# ----- byot_license_required: SAM3 / SAM3.1 / DINOv3 (gated custom license) ---
_SAM3_URL = "https://huggingface.co/facebook/sam3"
_SAM31_URL = "https://huggingface.co/facebook/sam3.1"
_SAM3_LIC = "SAM License (Meta custom, gated)"
for _mid in (
    "sam3-base",
    "sam3-image",
    "sam3-video",
    "sam3-text-prompt",
    "sam3-visual-prompt",
    "sam3-exemplar-prompt",
    "sam3-open-vocabulary",
    "sam3-tracking",
):
    _ROWS.append(
        _byot(
            _mid,
            "sam3",
            code=_SAM3_LIC,
            weights=_SAM3_LIC,
            hf_repo="facebook/sam3",
            upstream=_SAM3_URL,
            notes="Gated custom SAM License (not Apache). Provenance flagged "
            "unverified post-cutoff; commercial use depends on accepted terms.",
        )
    )
for _mid in (
    "sam3.1-base",
    "sam3.1-image",
    "sam3.1-video",
    "sam3.1-open-vocabulary",
    "sam3.1-text-prompt",
    "sam3.1-visual-prompt",
    "sam3.1-real-time-tracking",
):
    _ROWS.append(
        _byot(
            _mid,
            "sam3.1",
            code=_SAM3_LIC,
            weights=_SAM3_LIC,
            hf_repo="facebook/sam3.1",
            upstream=_SAM31_URL,
            notes="Gated custom SAM License (not Apache). Drop-in for SAM 3.",
        )
    )
_DINOV3_LIC = "DINOv3 License (Meta custom, gated)"
_dinov3 = {
    "dinov3-vits16": ("facebook/dinov3-vits16-pretrain-lvd1689m", "ViT-S/16 ~21M"),
    "dinov3-vitb16": ("facebook/dinov3-vitb16-pretrain-lvd1689m", "ViT-B/16 ~86M"),
    "dinov3-vitl16": ("facebook/dinov3-vitl16-pretrain-lvd1689m", "ViT-L/16 ~300M"),
    "dinov3-vit7b16": (
        "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "ViT-7B/16 ~6.8B (large footprint)",
    ),
    "dinov3-convnext-tiny": (
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "ConvNeXt-Tiny ~29M",
    ),
    "dinov3-convnext-small": (
        "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        "ConvNeXt-Small ~50M",
    ),
    "dinov3-convnext-base": (
        "facebook/dinov3-convnext-base-pretrain-lvd1689m",
        "ConvNeXt-Base ~119M",
    ),
    "dinov3-convnext-large": (
        "facebook/dinov3-convnext-large-pretrain-lvd1689m",
        "ConvNeXt-Large ~198M",
    ),
}
for _mid, (_repo, _desc) in _dinov3.items():
    _ROWS.append(
        _byot(
            _mid,
            "dinov3",
            code=_DINOV3_LIC,
            weights=_DINOV3_LIC,
            hf_repo=_repo,
            upstream=f"https://huggingface.co/{_repo}",
            notes=f"{_desc}. Custom DINOv3 License: commercial use permitted "
            f"with 'Built with DINOv3' attribution + acceptable-use + "
            f"no-compete-training conditions. Gated; BYOT only.",
        )
    )

# ----- external_api_only_terms_required: GroundingDINO 1.5/1.6, DINO-X --------
_ROWS += [
    _api(
        "grounding-dino-1.5",
        "grounding-dino",
        code="Apache-2.0 (client SDK only)",
        provider_env="DDS_API_KEY",
        upstream=f"{_GH}/IDEA-Research/Grounding-DINO-1.5-API",
        notes="Cloud-only (DeepDataSpace); weights proprietary.",
    ),
    _api(
        "grounding-dino-1.5-pro",
        "grounding-dino",
        code="Apache-2.0 (client SDK only)",
        provider_env="DDS_API_KEY",
        upstream=f"{_GH}/IDEA-Research/Grounding-DINO-1.5-API",
        notes="API-only, no weights. Commercial via paid quota + API ToS.",
    ),
    _api(
        "grounding-dino-1.6-pro",
        "grounding-dino",
        code="Apache-2.0 (client SDK only)",
        provider_env="DDS_API_KEY",
        upstream=f"{_GH}/IDEA-Research/Grounding-DINO-1.5-API",
        notes="SOTA variant. API-only, no weights.",
    ),
    _api(
        "dino-x-api",
        "dino-x",
        upstream=f"{_GH}/IDEA-Research/DINO-X-API",
        notes="Meta-entry for the DINO-X suite. DINOX_API_KEY; paid quota.",
    ),
    _api("dino-x-detection", "dino-x", upstream=f"{_GH}/IDEA-Research/DINO-X-API"),
    _api("dino-x-segmentation", "dino-x", upstream=f"{_GH}/IDEA-Research/DINO-X-API"),
    _api("dino-x-phrase-grounding", "dino-x", upstream=f"{_GH}/IDEA-Research/DINO-X-API"),
    _api("dino-x-counting", "dino-x", upstream=f"{_GH}/IDEA-Research/DINO-X-API"),
    _api("dino-x-region-captioning", "dino-x", upstream=f"{_GH}/IDEA-Research/DINO-X-API"),
]

# ----- noncommercial_restricted ----------------------------------------------
_ROWS += [
    _noncommercial(
        "edge-sam",
        "efficient-sam",
        code="NTU S-Lab License 1.0",
        weights="NTU S-Lab License 1.0 (non-commercial)",
        dataset_risk="high",
        upstream=f"{_GH}/chongzhou96/EdgeSAM",
        notes="Authors' own license is non-commercial. Commercial only via NTUitive.",
        aliases=("edgesam",),
    ),
    _noncommercial(
        "locate-anything-3b",
        "vlm-grounding",
        code="NVIDIA License",
        weights="NVIDIA License (non-commercial)",
        gated=True,
        hf_repo="nvidia/LocateAnything-3B",
        upstream="https://huggingface.co/nvidia/LocateAnything-3B",
        notes="Combined weights explicitly non-commercial. BYOT/local-cache only.",
        aliases=("locateanything-3b",),
    ),
    _noncommercial(
        "describe-anything-3b",
        "vlm-grounding",
        code="NVIDIA License",
        weights="NVIDIA License (non-commercial)",
        gated=True,
        hf_repo="nvidia/DAM-3B",
        upstream="https://huggingface.co/nvidia/DAM-3B",
        notes="NVIDIA Describe Anything Model (DAM): non-commercial research license.",
        aliases=("dam", "dam-3b"),
    ),
    _noncommercial(
        "medsam2",
        "sam",
        code="Apache-2.0 (code)",
        weights="non-commercial (medical dataset provenance)",
        dataset_risk="high",
        upstream=f"{_GH}/bowang-lab/MedSAM2",
        notes="Code Apache-2.0 but weights trained on medical datasets with "
        "non-commercial / research-only terms. Sidecar; research-only.",
    ),
    _noncommercial(
        "depth-anything-v2-large",
        "depth-anything",
        code="Apache-2.0 (code)",
        weights="CC-BY-NC-4.0",
        hf_repo="depth-anything/Depth-Anything-V2-Large",
        upstream=f"{_GH}/DepthAnything/Depth-Anything-V2",
        notes="Larger Depth-Anything V2 weights are CC-BY-NC-4.0 (non-commercial). "
        "Only the *small* V2 checkpoint is Apache-2.0 (-> commercial_safe_core).",
        aliases=("depthanything-large", "depth-anything-v2-base"),
    ),
    _noncommercial(
        "simpleclick",
        "interactive-seg",
        code="MIT (code)",
        weights="encumbered (MAE CC-BY-NC backbone)",
        dataset_risk="high",
        upstream=f"{_GH}/uncbiag/SimpleClick",
        notes="MIT code but all backbones use Meta MAE (CC-BY-NC). Published "
        "weights effectively non-commercial. Retrain remediation possible.",
    ),
    _noncommercial(
        "focalclick",
        "interactive-seg",
        code="MIT (code)",
        weights="encumbered (NVIDIA SegFormer backbone, non-commercial)",
        upstream=f"{_GH}/yzluka/FocalClick",
        notes="SegFormer backbone is NVIDIA non-commercial. Published SegFormer "
        "variant weights need NVIDIA commercial licensing.",
    ),
]

# ----- enterprise_license_required: AGPL / enterprise ------------------------
_ROWS += [
    _enterprise(
        "fastsam-s",
        "fastsam",
        code="Apache-2.0 (CASIA repo)",
        weights="AGPL-3.0 coupling (ultralytics YOLOv8-seg dependency)",
        upstream=f"{_GH}/CASIA-IVA-Lab/FastSAM",
        notes="FastSAM is a YOLOv8-seg model with an ultralytics AGPL-3.0 runtime "
        "dependency + 2% SA-1B subset. Inconsistent to ship as core while "
        "yolov8-seg is excluded. Enterprise/AGPL review required.",
    ),
    _enterprise(
        "fastsam-x",
        "fastsam",
        code="Apache-2.0 (CASIA repo)",
        weights="AGPL-3.0 coupling (ultralytics YOLOv8x-seg dependency)",
        upstream=f"{_GH}/CASIA-IVA-Lab/FastSAM",
    ),
    _enterprise(
        "yolov8-seg",
        "ultralytics",
        code="AGPL-3.0",
        weights="AGPL-3.0",
        upstream=f"{_GH}/ultralytics/ultralytics",
        notes="AGPL-3.0 copyleft; commercial needs Ultralytics Enterprise License.",
    ),
    _enterprise(
        "yolo11-seg",
        "ultralytics",
        code="AGPL-3.0",
        weights="AGPL-3.0",
        upstream=f"{_GH}/ultralytics/ultralytics",
        notes="AGPL-3.0 copyleft; commercial needs Ultralytics Enterprise License.",
    ),
]

# ----- legal_review_required: provenance unclear -----------------------------
_ROWS += [
    _legal(
        "hq-sam",
        "sam-hq",
        code="Apache-2.0 (code)",
        weights="Apache-2.0 weights / HQSeg-44K dataset partly NC",
        upstream=f"{_GH}/SysCV/sam-hq",
        notes="HQSeg-44K includes ThinObject-5K (CC-BY-NC) + DIS5K (NC ToU). "
        "Dataset-to-weights inheritance unsettled.",
        aliases=("sam-hq",),
    ),
    _legal(
        "light-hq-sam",
        "sam-hq",
        code="Apache-2.0 (code)",
        weights="Apache-2.0 weights / HQSeg-44K dataset partly NC",
        upstream=f"{_GH}/SysCV/sam-hq",
        notes="TinyViT backbone; same HQSeg-44K dataset concern as hq-sam.",
    ),
    _legal(
        "sam-hq2",
        "sam-hq",
        code="Apache-2.0 (code)",
        weights="Apache-2.0 weights / HQSeg-44K dataset partly NC",
        upstream=f"{_GH}/SysCV/sam-hq",
        notes="HQ-SAM-2 (image+video). 'hq-sam2' is an alias of this canonical id.",
        aliases=("hq-sam2",),
    ),
    _legal(
        "tinysam",
        "efficient-sam",
        code="Apache-2.0",
        weights="Apache-2.0",
        dataset_risk="medium",
        upstream=f"{_GH}/xinghaochen/TinySAM",
        notes="Apache-2.0 weights; SA-1B research-only subset behind distillation. "
        "Held at legal_review per v3.8 conservative known-state.",
    ),
    _legal(
        "q-tinysam",
        "efficient-sam",
        code="Apache-2.0",
        weights="Apache-2.0",
        dataset_risk="medium",
        upstream=f"{_GH}/xinghaochen/TinySAM",
        notes="W8A8-quantized TinySAM; same SA-1B provenance question.",
    ),
    _legal(
        "clickseg",
        "interactive-seg",
        code="MIT (code)",
        weights="mixed (CDNet/HRNet permissive; FocalClick/SegFormer NC)",
        upstream=f"{_GH}/alibaba/ClickSEG",
        notes="Mixed repo: CDNet/HRNet variant potentially usable; the FocalClick/"
        "SegFormer variant inherits NVIDIA non-commercial. Scope review to checkpoint.",
        aliases=("clickseg_alibaba",),
    ),
    _legal(
        "oneformer",
        "oneformer",
        code="MIT (code)",
        weights="MIT weights / training-data review",
        upstream=f"{_GH}/SHI-Labs/OneFormer",
        notes="Code MIT; some checkpoints trained on datasets needing provenance review.",
        aliases=("oneformer-convnext-large", "oneformer-dinat-large"),
    ),
    _legal(
        "internimage",
        "internimage",
        code="MIT (code)",
        weights="MIT weights / DCNv3 + dataset review",
        upstream=f"{_GH}/OpenGVLab/InternImage",
        notes="Code MIT; large-scale pretraining provenance + DCNv3 custom-op review.",
    ),
    _legal(
        "medsam",
        "sam",
        code="Apache-2.0 (code)",
        weights="Apache-2.0 weights / medical dataset provenance",
        upstream=f"{_GH}/bowang-lab/MedSAM",
        notes="MedSAM v1 code Apache-2.0; medical training-data provenance review before "
        "commercial clinical use.",
    ),
    # RF-DETR-Seg XL/2XL: v37 research said the *seg* XL/2XL are Apache-2.0 (distinct
    # from the PML-1.0 detection-XL); v3.8 known-state flags them for enterprise terms
    # verification. Held at legal_review pending confirmation of current Roboflow terms.
    _legal(
        "rfdetr-seg-xl",
        "rf-detr",
        code="Apache-2.0 (per v3.7 research)",
        weights="Apache-2.0 (seg) — verify not PML-1.0 detection-XL",
        dataset_risk="low",
        upstream=f"{_GH}/roboflow/rf-detr",
        notes="CONTESTED: v3.7 adversarial research found seg-XL is Apache-2.0 and does "
        "NOT need rfdetr_plus (unlike detection-XL PML-1.0). v3.8 known-state asks to "
        "verify current terms. Held at legal_review until current Roboflow terms confirm.",
    ),
    _legal(
        "rfdetr-seg-2xl",
        "rf-detr",
        code="Apache-2.0 (per v3.7 research)",
        weights="Apache-2.0 (seg) — verify not PML-1.0 detection-2XL",
        dataset_risk="low",
        upstream=f"{_GH}/roboflow/rf-detr",
        notes="CONTESTED: same as rfdetr-seg-xl. Held at legal_review pending current terms.",
    ),
]

# ----- not_released_or_unverifiable ------------------------------------------
_ROWS += [
    _not_released(
        "grounding-dino-2",
        "grounding-dino",
        notes="No official source found (GitHub/HF/arXiv/DeepDataSpace) as of audit.",
        upstream="",
    ),
]

# --------------------------------------------------------------------------- #
# Index + public helpers
# --------------------------------------------------------------------------- #
POLICY: dict[str, ModelPolicy] = {}
_ALIAS_INDEX: dict[str, str] = {}
for _row in _ROWS:
    if _row.model_id in POLICY:
        raise ValueError(f"duplicate model_id in POLICY: {_row.model_id}")
    POLICY[_row.model_id] = _row
    for _a in _row.aliases:
        _ALIAS_INDEX[_a] = _row.model_id


def resolve_model_id(model_id: str) -> str:
    """Map an alias to its canonical model_id (identity if already canonical)."""
    if model_id in POLICY:
        return model_id
    return _ALIAS_INDEX.get(model_id, model_id)


def get_policy(model_id: str) -> ModelPolicy | None:
    """Return the :class:`ModelPolicy` for a model id (or alias), else None."""
    return POLICY.get(resolve_model_id(model_id))


def iter_policies():
    """Iterate canonical policies in declaration order."""
    return iter(_ROWS)


def matrix_rows() -> list[dict[str, object]]:
    """All policy rows as flat dicts (CSV/report generation)."""
    return [r.as_row() for r in _ROWS]
