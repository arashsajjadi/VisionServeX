# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Authoritative, machine-readable model-license policy — the single source of truth.

VisionServeX core is **commercial-safe by default**: a model is only treated as
commercial-safe when its *code AND weights* are verified permissive. Restricted
models (research-only, non-commercial, AGPL/copyleft, legal-review, BYO-license)
are blocked by default and may be used only through an explicit, acknowledged
research/BYO pathway.

This module reconciles the two underlying sources — the curated licence matrix
(:mod:`visionservex.licensing.policy`) and the registry (`models.yaml`) — into one
``ModelLicensePolicy`` view, and exposes the helpers + the construction gate used
by the Python API, the CLI, the registry, docs, and tests.

It imports no heavy/optional dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from visionservex.exceptions import (
    ModelAcknowledgementRequiredError,
    ModelAGPLRestrictedError,
    ModelLicenseRestrictedError,
    ModelLicenseReviewRequiredError,
    ModelNotCommercialSafeError,
    ModelRequiresBYOCheckpointError,
    ModelUseModeNotAllowedError,
)
from visionservex.licensing.policy import get_policy as _get_curated_policy
from visionservex.registry import default_registry

# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
COMMERCIAL_STATUSES = (
    "commercial_safe",
    "noncommercial_restricted",
    "research_only",
    "agpl_restricted",
    "legal_review_required",
    "byo_license_only",
    "framework_only",
    "unknown",
)
PACKAGE_TIERS = ("core", "optional_commercial_safe", "research", "byo", "external", "hidden")
USE_MODES = ("commercial", "research", "education", "internal_evaluation", "byo_license")
FINETUNING_STATUSES = (
    "supported",
    "dry_run_only",
    "external_only",
    "not_trainable_by_design",
    "unknown",
)
CLI_WARNING_LEVELS = ("none", "info", "warning", "blocking")

#: Statuses that are NOT commercial-safe (excluded from the commercial-safe set,
#: assert_commercial_safe fails, predict CLI refuses by default).
_RESTRICTED_STATUSES = frozenset(
    {
        "noncommercial_restricted",
        "research_only",
        "agpl_restricted",
        "legal_review_required",
        "byo_license_only",
        "unknown",
    }
)

#: Statuses with a HARD upstream license restriction — blocked at VisionModel
#: construction unless explicitly acknowledged. ``legal_review_required`` is a
#: SOFT block (constructible for introspection/use, but never commercial-safe):
#: its code is typically permissive and only weight/dataset provenance is unverified.
_HARD_RESTRICTED_STATUSES = frozenset(
    {
        "noncommercial_restricted",
        "research_only",
        "agpl_restricted",
        "byo_license_only",
        "unknown",
    }
)

#: Map the curated licence matrix bucket -> public commercial_status + tier.
_FINAL_POLICY_MAP: dict[str, tuple[str, str]] = {
    "commercial_safe_core": ("commercial_safe", "core"),
    "noncommercial_restricted": ("noncommercial_restricted", "research"),
    "research_only": ("research_only", "research"),
    "enterprise_license_required": ("agpl_restricted", "external"),
    "legal_review_required": ("legal_review_required", "hidden"),
    "byot_license_required": ("byo_license_only", "byo"),
    "external_api_only_terms_required": ("legal_review_required", "external"),
    "not_released_or_unverifiable": ("unknown", "hidden"),
}

#: Models re-verified against official sources in this work (UTC date).
_VERIFIED: dict[str, dict[str, str]] = {
    "medsam2": {
        "date": "2026-06-22",
        "source": "https://huggingface.co/wanglab/MedSAM2",
        "status": "research_only",
        "note": "HF model card: 'The model weights can only be used for research and "
        "education purposes.' Code Apache-2.0; weights research/education only.",
    },
    "medsam": {
        "date": "2026-06-22",
        "source": "https://github.com/bowang-lab/MedSAM",
        "note": "Code Apache-2.0; medical-dataset weight provenance → legal review before commercial use.",
    },
    "sam2.1-hiera-tiny": {
        "date": "2026-06-22",
        "source": "https://github.com/facebookresearch/sam2",
        "note": "SAM 2.1 code + weights Apache-2.0 (SA-1B research-only dataset provenance documented).",
    },
}

#: Research-only weights override (forces research_only even if curated bucket is generic).
_RESEARCH_ONLY_IDS = frozenset({"medsam2"})

#: status -> default package tier for curated-override rows.
_STATUS_TIER_MAP = {
    "commercial_safe": "optional_commercial_safe",
    "research_only": "research",
    "noncommercial_restricted": "research",
    "agpl_restricted": "external",
    "legal_review_required": "hidden",
    "byo_license_only": "byo",
    "unknown": "hidden",
}

#: Curated per-model / per-family policy overrides, verified against OFFICIAL sources
#: (GitHub LICENSE / HF model card) on 2026-06-22. An id key beats a family key; both
#: beat the registry-licence fallback. This is where flagship models are made strict
#: WITHOUT bloating the legacy licence matrix. Keep the verification date current.
_GH = "https://github.com"
_CURATED_OVERRIDES: dict[str, dict[str, str]] = {
    # ---- verified commercial-safe flagships (code AND weights permissive) ----
    "dfine": {
        "status": "commercial_safe",
        "code": "Apache-2.0",
        "weights": "Apache-2.0",
        "source": f"{_GH}/Peterande/D-FINE",
        "note": "Apache-2.0 code+weights (gh license). Trained on Objects365+COCO — dataset provenance noted; weights released Apache-2.0 by the authors.",
    },
    "rfdetr": {
        "status": "commercial_safe",
        "code": "Apache-2.0",
        "weights": "Apache-2.0",
        "source": f"{_GH}/roboflow/rf-detr",
        "note": "Apache-2.0 (gh license). Applies to standard RF-DETR variants; XL/2XL handled separately.",
    },
    "grounding-dino": {
        "status": "commercial_safe",
        "code": "Apache-2.0",
        "weights": "Apache-2.0",
        "source": f"{_GH}/IDEA-Research/GroundingDINO",
        "note": "Open Grounding DINO is Apache-2.0 (gh license). API-gated 1.5/1.6/DINO-X are classified separately.",
    },
    "florence-2": {
        "status": "commercial_safe",
        "code": "MIT",
        "weights": "MIT",
        "source": "https://huggingface.co/microsoft/Florence-2-base",
        "note": "Microsoft Florence-2 released under MIT (model card).",
    },
    "clip": {
        "status": "commercial_safe",
        "code": "MIT",
        "weights": "MIT",
        "source": "https://huggingface.co/openai/clip-vit-base-patch32",
        "note": "OpenAI CLIP — MIT.",
    },
    "siglip": {
        "status": "commercial_safe",
        "code": "Apache-2.0",
        "weights": "Apache-2.0",
        "source": "https://huggingface.co/google/siglip-base-patch16-224",
        "note": "Google SigLIP — Apache-2.0.",
    },
    "siglip2": {
        "status": "commercial_safe",
        "code": "Apache-2.0",
        "weights": "Apache-2.0",
        "source": "https://huggingface.co/google/siglip2-base-patch16-224",
        "note": "Google SigLIP2 — Apache-2.0.",
    },
    "owlv2": {
        "status": "commercial_safe",
        "code": "Apache-2.0",
        "weights": "Apache-2.0",
        "source": "https://huggingface.co/google/owlv2-base-patch16",
        "note": "Google OWLv2 — Apache-2.0.",
    },
    "owlvit": {
        "status": "commercial_safe",
        "code": "Apache-2.0",
        "weights": "Apache-2.0",
        "source": "https://huggingface.co/google/owlvit-base-patch32",
        "note": "Google OWL-ViT — Apache-2.0.",
    },
    "swinv2": {
        "status": "commercial_safe",
        "code": "MIT",
        "weights": "MIT",
        "source": f"{_GH}/microsoft/Swin-Transformer",
        "note": "Microsoft Swin Transformer V2 — MIT.",
    },
    "maxvit": {
        "status": "commercial_safe",
        "code": "Apache-2.0",
        "weights": "Apache-2.0",
        "source": f"{_GH}/google-research/maxvit",
        "note": "Google MaxViT — Apache-2.0 (timm weights).",
    },
    "rtmpose": {
        "status": "commercial_safe",
        "code": "Apache-2.0",
        "weights": "Apache-2.0",
        "source": f"{_GH}/open-mmlab/mmpose",
        "note": "RTMPose (OpenMMLab MMPose) — Apache-2.0.",
    },
    "grounded-sam": {
        "status": "commercial_safe",
        "code": "Apache-2.0",
        "weights": "Apache-2.0",
        "source": f"{_GH}/IDEA-Research/Grounded-Segment-Anything",
        "note": "Composes Grounding DINO (Apache-2.0) + SAM (Apache-2.0); both permissive.",
    },
    "grounded-sam2": {
        "status": "commercial_safe",
        "code": "Apache-2.0",
        "weights": "Apache-2.0",
        "source": f"{_GH}/IDEA-Research/Grounded-SAM-2",
        "note": "Composes Grounding DINO (Apache-2.0) + SAM 2 (Apache-2.0); both permissive.",
    },
    # ---- downgrades: license traps caught (NOT commercial-safe) ----
    "convnextv2": {
        "status": "legal_review_required",
        "code": "MIT/CC-BY-NC (conflict)",
        "weights": "CC-BY-NC-4.0 (upstream) vs Apache-2.0 (HF tag)",
        "source": f"{_GH}/facebookresearch/ConvNeXt-V2",
        "note": "CONFLICT: upstream LICENSE is CC-BY-NC-4.0 (non-commercial) while the HF model card tags apache-2.0. Stricter interpretation wins → NOT commercial-safe pending resolution.",
    },
    "rfdetr-seg-xlarge": {
        "status": "legal_review_required",
        "code": "Apache-2.0 (code)",
        "weights": "verify (XL enterprise terms)",
        "source": f"{_GH}/roboflow/rf-detr",
        "note": "RF-DETR-Seg XL: enterprise-terms verification pending; not confirmed Apache-2.0 for this variant. Not commercial-safe until resolved.",
    },
    "rfdetr-seg-2xlarge": {
        "status": "legal_review_required",
        "code": "Apache-2.0 (code)",
        "weights": "verify (2XL enterprise terms)",
        "source": f"{_GH}/roboflow/rf-detr",
        "note": "RF-DETR-Seg 2XL: enterprise-terms verification pending; not commercial-safe until resolved.",
    },
    "oneformer-swin-large": {
        "status": "legal_review_required",
        "code": "MIT (code)",
        "weights": "provenance review",
        "source": f"{_GH}/SHI-Labs/OneFormer",
        "note": "OneFormer weights provenance review (consistency with the convnext/dinat-large variants). Not commercial-safe until resolved.",
    },
}


def _lookup_override(model_id: str, family: str) -> dict[str, str] | None:
    """Curated override for a model: exact id wins over family."""
    if model_id in _CURATED_OVERRIDES:
        return _CURATED_OVERRIDES[model_id]
    if family and family in _CURATED_OVERRIDES:
        return _CURATED_OVERRIDES[family]
    return None


#: Permissive licence prefixes (code+weights) used for no-curated-row fallback.
_PERMISSIVE = ("apache", "mit", "bsd")
_COPYLEFT = ("agpl", "gpl-3", "gplv3", "gpl-2", "lgpl")
_NONCOMMERCIAL_MARKERS = ("cc-by-nc", "noncommercial", "non-commercial", "-nc", "research")

ACKNOWLEDGEMENT_TEXT = (
    "I understand that this model is not commercial-safe by default. I confirm that I "
    "have the right to use the model weights for this use case. Outputs are AI-generated "
    "suggestions only and are not for diagnosis, treatment, or clinical decision-making."
)


# --------------------------------------------------------------------------- #
# Policy view
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelLicensePolicy:
    """Unified, machine-readable license policy for one model id."""

    model_id: str
    display_name: str
    provider: str
    family: str
    task: str
    modality: str
    code_license: str
    weights_license: str
    commercial_status: str
    default_package_tier: str
    allowed_use_modes: tuple[str, ...]
    auto_download_allowed: bool
    requires_acknowledgement: bool
    requires_byo_checkpoint: bool
    allows_finetuning: bool
    finetuning_status: str
    default_enabled: bool
    not_for_diagnosis_required: bool
    cli_warning_level: str
    source_url: str
    last_verified_date: str
    policy_notes: str
    in_registry: bool = True
    policy_source: str = "registry_derived"  # curated_matrix | curated_override | registry_derived
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_commercial_safe(self) -> bool:
        return self.commercial_status == "commercial_safe"

    @property
    def is_restricted(self) -> bool:
        return self.commercial_status in _RESTRICTED_STATUSES

    @property
    def is_hard_restricted(self) -> bool:
        """Hard upstream restriction — blocked at construction without acknowledgement."""
        return self.commercial_status in _HARD_RESTRICTED_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "provider": self.provider,
            "family": self.family,
            "task": self.task,
            "modality": self.modality,
            "code_license": self.code_license,
            "weights_license": self.weights_license,
            "commercial_status": self.commercial_status,
            "default_package_tier": self.default_package_tier,
            "allowed_use_modes": list(self.allowed_use_modes),
            "auto_download_allowed": self.auto_download_allowed,
            "requires_acknowledgement": self.requires_acknowledgement,
            "requires_byo_checkpoint": self.requires_byo_checkpoint,
            "allows_finetuning": self.allows_finetuning,
            "finetuning_status": self.finetuning_status,
            "default_enabled": self.default_enabled,
            "not_for_diagnosis_required": self.not_for_diagnosis_required,
            "cli_warning_level": self.cli_warning_level,
            "source_url": self.source_url,
            "last_verified_date": self.last_verified_date,
            "policy_notes": self.policy_notes,
            "in_registry": self.in_registry,
            "policy_source": self.policy_source,
            "is_commercial_safe": self.is_commercial_safe,
        }


def _modality_for_task(task: str) -> str:
    if task in ("segment", "foundation_segment", "grounded_segment"):
        return "segmentation"
    if task in ("detect", "open_vocab_detect", "obb"):
        return "detection"
    if task == "classify":
        return "classification"
    if task == "embed":
        return "embedding"
    if task == "vlm":
        return "vision_language"
    return task or "image"


def _allowed_modes(status: str) -> tuple[str, ...]:
    if status == "commercial_safe":
        return USE_MODES
    if status == "byo_license_only":
        return ("byo_license", "research", "internal_evaluation")
    if status in ("research_only", "noncommercial_restricted"):
        return ("research", "education", "internal_evaluation")
    if status == "agpl_restricted":
        return ("research", "internal_evaluation", "byo_license")
    if status == "legal_review_required":
        return ("research", "internal_evaluation")
    if status == "framework_only":
        return USE_MODES
    return ("research", "internal_evaluation")


def _is_medical(family: str, task: str, model_id: str) -> bool:
    fam = (family or "").lower()
    mid = model_id.lower()
    return (
        "medsam" in mid
        or "medsam" in fam
        or "nnunet" in mid
        or "totalseg" in mid
        or "monai" in mid
        or "vista" in mid
        or "unetr" in fam
    )


def _classify_license_text(text: str) -> str:
    t = (text or "").lower()
    if any(m in t for m in _COPYLEFT):
        return "copyleft"
    if any(m in t for m in _NONCOMMERCIAL_MARKERS):
        return "noncommercial"
    if any(p in t for p in _PERMISSIVE):
        return "permissive"
    return "unknown"


def get_model_policy(model_id: str) -> ModelLicensePolicy:
    """Return the unified license policy for ``model_id`` (registry or known id).

    Fail-closed: a model with no curated row and a non-permissive / uncertain
    registry licence is reported as ``legal_review_required`` (not commercial-safe).
    """
    reg = default_registry()
    entry = reg.get(model_id) if reg.has(model_id) else None
    curated = _get_curated_policy(model_id)

    display = getattr(entry, "display_name", model_id) if entry else model_id
    family = getattr(entry, "family", "") if entry else (curated.family if curated else "")
    task = getattr(entry, "task", "") if entry else ""
    reg_license = getattr(entry, "license", "") if entry else ""
    license_uncertain = bool(getattr(entry, "license_uncertain", False)) if entry else False

    verified = _VERIFIED.get(model_id, {})
    override = _lookup_override(model_id, family)
    last_verified = ""
    policy_source = "registry_derived"

    if curated is not None:
        status, tier = _FINAL_POLICY_MAP.get(
            curated.final_policy, ("legal_review_required", "hidden")
        )
        code_license = curated.code_license or reg_license or "unknown"
        weights_license = curated.weights_license or "unknown"
        source = curated.upstream_url or curated.hf_repo or ""
        notes = curated.notes or ""
        auto_dl = bool(curated.can_auto_download)
        policy_source = "curated_matrix"
    elif override is not None:
        # Verified per-model/per-family policy (official source, dated).
        status = override["status"]
        tier = _STATUS_TIER_MAP.get(status, "hidden")
        code_license = override.get("code", reg_license or "unknown")
        weights_license = override.get("weights", "unknown")
        source = override.get("source", "")
        notes = override.get("note", "")
        auto_dl = False
        last_verified = "2026-06-22"
        policy_source = "curated_override"
    else:
        # No curated row → derive from the registry licence, fail-closed on doubt.
        lclass = _classify_license_text(reg_license)
        if license_uncertain or lclass == "unknown":
            status, tier = "legal_review_required", "hidden"
        elif lclass == "copyleft":
            status, tier = "agpl_restricted", "external"
        elif lclass == "noncommercial":
            status, tier = "noncommercial_restricted", "research"
        else:  # permissive code+weights, nothing flagged → commercial-safe
            status, tier = "commercial_safe", "optional_commercial_safe"
        code_license = reg_license or "unknown"
        weights_license = reg_license or "unknown"
        source = getattr(entry, "upstream_url", "") if entry else ""
        notes = getattr(entry, "commercial_use_notes", "") if entry else ""
        auto_dl = bool(getattr(entry, "auto_download", False)) if entry else False
        policy_source = "registry_derived"

    # Research-only weights override (e.g. MedSAM2 model card).
    if model_id in _RESEARCH_ONLY_IDS:
        status, tier = "research_only", "research"

    medical = _is_medical(family, task, model_id)
    hard_restricted = status in _HARD_RESTRICTED_STATUSES
    requires_byo = status == "byo_license_only"

    warning = "none"
    if status == "commercial_safe":
        warning = "none"
    elif status in ("research_only", "noncommercial_restricted", "agpl_restricted"):
        warning = "blocking"
    elif status in ("legal_review_required", "byo_license_only", "unknown"):
        warning = "warning"

    # Fine-tuning truth (medical foundation segmenters are not trainable by design).
    if medical and ("medsam" in model_id):
        ft_status, allows_ft = "not_trainable_by_design", False
    else:
        ft_status, allows_ft = "unknown", False

    return ModelLicensePolicy(
        model_id=model_id,
        display_name=display,
        provider=(
            source.split("/")[3] if source.startswith("http") and source.count("/") > 3 else ""
        ),
        family=family,
        task=task,
        modality=_modality_for_task(task),
        code_license=code_license,
        weights_license=weights_license,
        commercial_status=status,
        default_package_tier=tier,
        allowed_use_modes=_allowed_modes(status),
        auto_download_allowed=bool(auto_dl and status == "commercial_safe"),
        requires_acknowledgement=hard_restricted,
        requires_byo_checkpoint=requires_byo,
        allows_finetuning=allows_ft,
        finetuning_status=ft_status,
        default_enabled=(status == "commercial_safe"),
        not_for_diagnosis_required=medical,
        cli_warning_level=warning,
        source_url=verified.get("source", source),
        last_verified_date=verified.get("date", last_verified),
        policy_notes=verified.get("note", notes),
        in_registry=entry is not None,
        policy_source=policy_source,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def list_models() -> list[str]:
    """All registry model ids (sorted)."""
    return sorted(e.id for e in default_registry().list())


def list_commercial_safe_models() -> list[str]:
    """Registry models whose code + weights are verified commercial-safe."""
    return sorted(m for m in list_models() if get_model_policy(m).is_commercial_safe)


#: Known restricted models that are not runtime-registry entries but are
#: classified by VisionServeX (research/BYO/blocked-by-default) and discoverable.
_KNOWN_NONREGISTRY_RESTRICTED = ("medsam2",)


def list_research_models() -> list[str]:
    """Models usable only for research/education/internal evaluation (incl. sidecars)."""
    candidates = set(list_models()) | set(_KNOWN_NONREGISTRY_RESTRICTED)
    return sorted(
        m
        for m in candidates
        if get_model_policy(m).commercial_status in ("research_only", "noncommercial_restricted")
    )


def list_byo_models() -> list[str]:
    """Models that require a user-supplied checkpoint/license (BYO)."""
    return sorted(
        m for m in list_models() if get_model_policy(m).commercial_status == "byo_license_only"
    )


def list_legal_review_models() -> list[str]:
    """Models pending legal review (not commercial-safe; ambiguous/unverified weights)."""
    return sorted(
        m for m in list_models() if get_model_policy(m).commercial_status == "legal_review_required"
    )


def policy_coverage() -> dict[str, Any]:
    """Machine-readable policy-coverage report (curated vs registry-derived)."""
    from collections import Counter

    ids = list_models()
    status: Counter = Counter()
    source: Counter = Counter()
    safe_source: Counter = Counter()
    registry_derived_safe: list[str] = []
    for m in ids:
        p = get_model_policy(m)
        status[p.commercial_status] += 1
        source[p.policy_source] += 1
        if p.is_commercial_safe:
            safe_source[p.policy_source] += 1
            if p.policy_source == "registry_derived":
                registry_derived_safe.append(m)
    commercial_safe = sum(1 for m in ids if get_model_policy(m).is_commercial_safe)
    curated_safe = safe_source["curated_matrix"] + safe_source["curated_override"]
    return {
        "total_models": len(ids),
        "by_commercial_status": dict(status),
        "by_policy_source": dict(source),
        "commercial_safe_total": commercial_safe,
        "commercial_safe_curated": curated_safe,
        "commercial_safe_registry_derived": safe_source["registry_derived"],
        "commercial_safe_curated_pct": round(100 * curated_safe / max(commercial_safe, 1), 1),
        "registry_derived_commercial_safe_ids": sorted(registry_derived_safe),
    }


def assert_commercial_safe(model_id: str) -> None:
    """Raise unless ``model_id`` is verified commercial-safe."""
    pol = get_model_policy(model_id)
    if not pol.is_commercial_safe:
        raise ModelNotCommercialSafeError(model_id, pol.commercial_status)


def explain_model_license(model_id: str) -> str:
    """Human-readable license explanation for ``model_id``."""
    p = get_model_policy(model_id)
    lines = [
        f"{model_id} — {p.display_name}",
        f"  commercial status : {p.commercial_status}"
        + ("  ✅ commercial-safe" if p.is_commercial_safe else "  ⚠ NOT commercial-safe"),
        f"  code license      : {p.code_license}",
        f"  weights license   : {p.weights_license}",
        f"  package tier      : {p.default_package_tier}",
        f"  allowed use modes : {', '.join(p.allowed_use_modes)}",
        f"  acknowledgement   : {'REQUIRED' if p.requires_acknowledgement else 'not required'}",
        f"  fine-tuning       : {p.finetuning_status}",
    ]
    if p.requires_byo_checkpoint:
        lines.append("  checkpoint        : BYO (supply your own)")
    if p.not_for_diagnosis_required:
        lines.append("  medical           : research/education only — NOT for diagnosis")
    if p.last_verified_date:
        lines.append(f"  last verified     : {p.last_verified_date} ({p.source_url})")
    if p.policy_notes:
        lines.append(f"  notes             : {p.policy_notes}")
    if not p.is_commercial_safe:
        lines.append("")
        lines.append("  To use this restricted model you must acknowledge:")
        lines.append(f'  "{ACKNOWLEDGEMENT_TEXT}"')
    return "\n".join(lines)


def check_use_allowed(
    model_id: str,
    *,
    use_mode: str = "commercial",
    acknowledged: bool = False,
    has_checkpoint: bool = False,
) -> ModelLicensePolicy:
    """Gate: raise a structured error if this use of ``model_id`` is not permitted.

    Commercial-safe models always pass. Restricted models require an explicit
    acknowledgement; the granular use_mode / BYO-checkpoint checks then apply.
    Returns the policy on success.
    """
    p = get_model_policy(model_id)
    if p.is_commercial_safe:
        return p

    # legal_review_required / soft-restricted: not commercial-safe (excluded from
    # the commercial-safe set, assert_commercial_safe fails, CLI predict refuses by
    # default), but constructible for introspection/research use. No hard block.
    if not p.is_hard_restricted:
        return p

    # Hard upstream restriction: acknowledgement is the master key for construction.
    if not acknowledged:
        if p.commercial_status == "agpl_restricted":
            raise ModelAGPLRestrictedError(model_id)
        if p.commercial_status in ("research_only", "noncommercial_restricted"):
            raise ModelAcknowledgementRequiredError(
                model_id, p.commercial_status, ACKNOWLEDGEMENT_TEXT
            )
        if p.commercial_status == "byo_license_only":
            raise ModelLicenseRestrictedError(
                model_id, p.commercial_status, hint="pass use_mode='byo_license' + checkpoint + ack"
            )
        if p.commercial_status == "unknown":
            raise ModelLicenseReviewRequiredError(model_id)
        raise ModelLicenseRestrictedError(model_id, p.commercial_status)

    # Acknowledged: validate use_mode + BYO checkpoint.
    if use_mode not in p.allowed_use_modes:
        raise ModelUseModeNotAllowedError(model_id, use_mode, p.allowed_use_modes)
    if p.requires_byo_checkpoint and use_mode == "byo_license" and not has_checkpoint:
        raise ModelRequiresBYOCheckpointError(model_id)
    return p


__all__ = [
    "ACKNOWLEDGEMENT_TEXT",
    "CLI_WARNING_LEVELS",
    "COMMERCIAL_STATUSES",
    "FINETUNING_STATUSES",
    "PACKAGE_TIERS",
    "USE_MODES",
    "ModelLicensePolicy",
    "assert_commercial_safe",
    "check_use_allowed",
    "explain_model_license",
    "get_model_policy",
    "list_byo_models",
    "list_commercial_safe_models",
    "list_legal_review_models",
    "list_models",
    "list_research_models",
    "policy_coverage",
]
