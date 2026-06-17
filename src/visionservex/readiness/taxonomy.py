# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Precise model-readiness taxonomy (v3.18).

The legacy ``model_capabilities()`` readiness was a four-value bucket
(``train-ready`` / ``inference-ready`` / ``catalog-only`` / ``blocked``).
That is honest but coarse: it cannot tell a gated BYOT model from a missing
dependency, nor a *live-verified* model from a merely *capability-derived* one.

This module defines the exact, machine-consumable readiness vocabulary that
Anastig (and any downstream UI) drives off. It is intentionally dependency-free
and pure: every function takes plain values and returns plain values, so it is
trivially unit-testable and free of import cycles with ``core.model``.

The cardinal rule, enforced by tests:

    No state contains ``READY`` *as a live promise* unless either
      (a) a live smoke test passed in this sprint  -> ``*_READY_LIVE``, or
      (b) it is explicitly flagged derived          -> ``*_DERIVED_NEEDS_LIVE_CONFIRMATION``.

The coarse legacy bucket is kept (``coarse_readiness``) and is a pure function
of the precise state, so the two can never silently drift.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# The 22 readiness states.
# --------------------------------------------------------------------------- #
TRAIN_READY_LIVE = "TRAIN_READY_LIVE"
TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION = "TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION"
INFERENCE_READY_LIVE = "INFERENCE_READY_LIVE"
INFERENCE_READY_DERIVED_NEEDS_LIVE_CONFIRMATION = "INFERENCE_READY_DERIVED_NEEDS_LIVE_CONFIRMATION"
EMBEDDING_READY_LIVE = "EMBEDDING_READY_LIVE"
SEGMENTATION_READY_LIVE = "SEGMENTATION_READY_LIVE"
OPEN_VOCAB_READY_LIVE = "OPEN_VOCAB_READY_LIVE"
VLM_READY_LIVE = "VLM_READY_LIVE"
CORRESPONDENCE_READY_LIVE = "CORRESPONDENCE_READY_LIVE"
GATED_TOKEN_REQUIRED = "GATED_TOKEN_REQUIRED"
LICENSE_BLOCKED = "LICENSE_BLOCKED"
NON_COMMERCIAL_BLOCKED = "NON_COMMERCIAL_BLOCKED"
CATALOG_ONLY_ENGINE_NOT_WIRED = "CATALOG_ONLY_ENGINE_NOT_WIRED"
CUSTOM_LOADER_REQUIRED = "CUSTOM_LOADER_REQUIRED"
DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
WEIGHTS_MISSING = "WEIGHTS_MISSING"
UPSTREAM_CRASH = "UPSTREAM_CRASH"
OOM_BLOCKED = "OOM_BLOCKED"
TASK_NOT_SUPPORTED = "TASK_NOT_SUPPORTED"
PARTIAL_IMPLEMENTATION_BLOCKED = "PARTIAL_IMPLEMENTATION_BLOCKED"
UNKNOWN_REVIEW_REQUIRED = "UNKNOWN_REVIEW_REQUIRED"

READINESS_STATES: frozenset[str] = frozenset(
    {
        TRAIN_READY_LIVE,
        TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION,
        INFERENCE_READY_LIVE,
        INFERENCE_READY_DERIVED_NEEDS_LIVE_CONFIRMATION,
        EMBEDDING_READY_LIVE,
        SEGMENTATION_READY_LIVE,
        OPEN_VOCAB_READY_LIVE,
        VLM_READY_LIVE,
        CORRESPONDENCE_READY_LIVE,
        GATED_TOKEN_REQUIRED,
        LICENSE_BLOCKED,
        NON_COMMERCIAL_BLOCKED,
        CATALOG_ONLY_ENGINE_NOT_WIRED,
        CUSTOM_LOADER_REQUIRED,
        DEPENDENCY_MISSING,
        WEIGHTS_MISSING,
        UPSTREAM_CRASH,
        OOM_BLOCKED,
        TASK_NOT_SUPPORTED,
        PARTIAL_IMPLEMENTATION_BLOCKED,
        UNKNOWN_REVIEW_REQUIRED,
    }
)

# Live task-specific "ready" states — the ONLY states a downstream UI may treat
# as usable-by-default. ``*_DERIVED_*`` is explicitly excluded.
LIVE_READY_STATES: frozenset[str] = frozenset(
    {
        TRAIN_READY_LIVE,
        INFERENCE_READY_LIVE,
        EMBEDDING_READY_LIVE,
        SEGMENTATION_READY_LIVE,
        OPEN_VOCAB_READY_LIVE,
        VLM_READY_LIVE,
        CORRESPONDENCE_READY_LIVE,
    }
)

# Per-task live inference state (train-ready is handled separately).
_LIVE_STATE_FOR_TASK: dict[str, str] = {
    "detect": INFERENCE_READY_LIVE,
    "obb": INFERENCE_READY_LIVE,
    "pose": INFERENCE_READY_LIVE,
    "classify": INFERENCE_READY_LIVE,
    "classification": INFERENCE_READY_LIVE,
    "embed": EMBEDDING_READY_LIVE,
    "embedding": EMBEDDING_READY_LIVE,
    "segment": SEGMENTATION_READY_LIVE,
    "foundation_segment": SEGMENTATION_READY_LIVE,
    "grounded_segment": SEGMENTATION_READY_LIVE,
    "open_vocab_detect": OPEN_VOCAB_READY_LIVE,
    "vlm": VLM_READY_LIVE,
    "correspondence": CORRESPONDENCE_READY_LIVE,
}


# --------------------------------------------------------------------------- #
# Coarse legacy bucket (kept for backward compatibility / existing tests).
# --------------------------------------------------------------------------- #
def coarse_readiness(state: str) -> str:
    """Map a precise readiness state to the legacy 4-value bucket.

    ``train-ready`` / ``inference-ready`` / ``catalog-only`` / ``blocked``.
    This is a *pure derivation* — the precise state is authoritative.
    """
    if state in (TRAIN_READY_LIVE, TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION):
        return "train-ready"
    if state in (
        INFERENCE_READY_LIVE,
        INFERENCE_READY_DERIVED_NEEDS_LIVE_CONFIRMATION,
        EMBEDDING_READY_LIVE,
        SEGMENTATION_READY_LIVE,
        OPEN_VOCAB_READY_LIVE,
        VLM_READY_LIVE,
        CORRESPONDENCE_READY_LIVE,
    ):
        return "inference-ready"
    if state in (CATALOG_ONLY_ENGINE_NOT_WIRED, CUSTOM_LOADER_REQUIRED, WEIGHTS_MISSING):
        return "catalog-only"
    # Everything else (legal blocks, gated, partial, crash, oom, dep-missing,
    # task-not-supported, unknown) is "blocked" in the coarse view.
    return "blocked"


# --------------------------------------------------------------------------- #
# Anastig visibility (what the product UI should do with this model).
# --------------------------------------------------------------------------- #
# show_train | show_inference | show_embedding | show_segmentation |
# show_token_required | hide | blocked_admin_only
_BLOCKED_TECHNICAL = frozenset(
    {
        CATALOG_ONLY_ENGINE_NOT_WIRED,
        CUSTOM_LOADER_REQUIRED,
        DEPENDENCY_MISSING,
        WEIGHTS_MISSING,
        UPSTREAM_CRASH,
        OOM_BLOCKED,
        TASK_NOT_SUPPORTED,
        PARTIAL_IMPLEMENTATION_BLOCKED,
    }
)
_BLOCKED_ADMIN = frozenset({LICENSE_BLOCKED, NON_COMMERCIAL_BLOCKED, UNKNOWN_REVIEW_REQUIRED})


def _inference_show_verb(task: str) -> str:
    if task in ("embed", "embedding"):
        return "show_embedding"
    if task in ("segment", "foundation_segment", "grounded_segment"):
        return "show_segmentation"
    return "show_inference"


def anastig_visibility(
    state: str,
    *,
    live_inference: bool = False,
    live_train: bool = False,
    task: str = "",
    legal_review: bool = False,
) -> str:
    """Return the Anastig visibility verb for a model.

    Visibility is driven by what is **actually usable now**, not just the
    headline readiness state. A model whose train lifecycle is only
    capability-derived but whose *inference* is live-verified (e.g. RF-DETR) is
    still ``show_inference`` — the train UI is hidden, the inference UI is not.

    Only live-verified capability is ever shown; ``*_DERIVED`` and every blocked
    state are hidden (or ``blocked_admin_only`` for legal/review states).
    """
    if state == GATED_TOKEN_REQUIRED:
        return "show_token_required"
    if state in _BLOCKED_ADMIN:
        return "blocked_admin_only"
    if state in _BLOCKED_TECHNICAL:
        return "hide"
    # Soft legal-review overlay hides otherwise-functional models from end users.
    if legal_review:
        return "blocked_admin_only"
    # Full train lifecycle live-verified -> train UI (implies inference too).
    if state == TRAIN_READY_LIVE or live_train:
        return "show_train"
    # Inference live-verified (covers the *_READY_LIVE inference states AND a
    # train-ready model whose inference is live but whose train is still derived).
    if live_inference or state in LIVE_READY_STATES:
        return _inference_show_verb(task)
    # Derived-but-not-live-verified, or anything else -> hidden until confirmed.
    return "hide"


# --------------------------------------------------------------------------- #
# License classification — the hard legal gate.
# --------------------------------------------------------------------------- #
# Copyleft families that VisionServeX forbids on any runtime/training path.
_COPYLEFT_MARKERS = ("agpl", "sspl", "gpl")  # 'gpl' minus the lgpl exception below
_NONCOMMERCIAL_MARKERS = (
    "non-commercial",
    "noncommercial",
    "non commercial",
    "cc-by-nc",
    "cc by nc",
    "by-nc",
    "research only",
    "research-only",
    "research purposes only",
    "nc-",
    "deci",  # YOLO-NAS / Deci proprietary non-commercial
)
# NB: openrail/creativeml are deliberately NOT permissive — they carry use
# restrictions and are classified ``custom_unknown`` (hidden until reviewed).
_PERMISSIVE_MARKERS = (
    "apache",
    "mit",
    "bsd",
    "isc",
    "cc0",
    "unlicense",
    "zlib",
    "python-2",
    "psf",
)
_CUSTOM_UNKNOWN_MARKERS = (
    "custom",
    "proprietary",
    "other",
    "see license",
    "see-license",
    "openrail",
    "creativeml",
    "research",
    "coqui",
    "llama",
    "gemma",
)


def classify_license(license_str: str | None) -> str:
    """Classify a license string into a legal class.

    Returns one of:
      ``permissive`` | ``copyleft`` | ``noncommercial`` | ``custom_unknown`` | ``unknown``

    Ordering matters: copyleft and non-commercial markers win over a stray
    permissive substring so e.g. "GPL with MIT test files" is still copyleft.
    """
    if not license_str or not str(license_str).strip():
        return "unknown"
    s = str(license_str).strip().lower()

    # Non-commercial first (some NC licenses are CC-derived and mention 'by').
    if any(m in s for m in _NONCOMMERCIAL_MARKERS):
        return "noncommercial"

    # Copyleft, with the LGPL carve-out (LGPL is weak copyleft / linkable).
    if "agpl" in s or "sspl" in s:
        return "copyleft"
    if "gpl" in s and "lgpl" not in s:
        return "copyleft"

    # Permissive.
    if any(m in s for m in _PERMISSIVE_MARKERS):
        return "permissive"

    # Custom / proprietary / use-restricted -> treat as unknown-ish (hidden).
    if any(m in s for m in _CUSTOM_UNKNOWN_MARKERS):
        return "custom_unknown"

    return "custom_unknown"


def is_copyleft(license_str: str | None) -> bool:
    return classify_license(license_str) == "copyleft"


def is_noncommercial(license_str: str | None) -> bool:
    return classify_license(license_str) == "noncommercial"


# --------------------------------------------------------------------------- #
# The central decision function.
# --------------------------------------------------------------------------- #
def compute_readiness_state(
    *,
    task: str,
    implementation_status: str,
    engine: str,
    engine_registered: bool,
    policy_bucket: str | None,
    license_class: str,
    unavailable_reason: str | None,
    train_ready: bool,
    inference_ready: bool,
    live_inference_verified: bool,
    live_train_verified: bool,
    live_inference_blocker: str | None = None,
) -> str:
    """Compute the precise readiness state for one model.

    Priority order (most-blocking first), so a model is described by its
    deepest *honest* blocker rather than an optimistic one:

      1. hard copyleft / enterprise license      -> LICENSE_BLOCKED
      2. non-commercial license                  -> NON_COMMERCIAL_BLOCKED
      3. BYOT-gated *with a real token path*      -> GATED_TOKEN_REQUIRED
      4. not-released / unverifiable weights      -> WEIGHTS_MISSING
      5. custom loader required                   -> CUSTOM_LOADER_REQUIRED
      6. partial implementation                   -> PARTIAL_IMPLEMENTATION_BLOCKED
      7. stub engine / not wired / external-api   -> CATALOG_ONLY_ENGINE_NOT_WIRED
      8. train-ready (live or derived)            -> TRAIN_READY_*
      9. inference-ready (live or derived)        -> *_READY_LIVE / INFERENCE_READY_DERIVED_*
     10. fallback                                 -> UNKNOWN_REVIEW_REQUIRED

    Note the deliberate distinction in step 3: a model is only promoted to
    ``GATED_TOKEN_REQUIRED`` when it carries a curated BYOT policy row (i.e. a
    real token-gated runtime exists). A merely ``requires_auth`` *stub* is not
    promoted — a token would not make a stub run — so it falls through to
    ``CATALOG_ONLY_ENGINE_NOT_WIRED`` (with ``requires_token`` still flagged
    truthfully elsewhere). This keeps the promise of ``GATED_TOKEN_REQUIRED``
    honest: "supply a token and it runs."
    """
    reason = (unavailable_reason or "").lower()

    # 1) Hard copyleft / enterprise legal block.
    if license_class == "copyleft" or policy_bucket == "enterprise_license_required":
        return LICENSE_BLOCKED

    # 2) Non-commercial.
    if license_class == "noncommercial" or policy_bucket in (
        "noncommercial_restricted",
        "excluded_from_core",
    ):
        return NON_COMMERCIAL_BLOCKED

    # 3) BYOT-gated with a real token-gated runtime path.
    if policy_bucket in ("byot_license_required", "auth_required_license_pending"):
        return GATED_TOKEN_REQUIRED

    # 4) Not released / unverifiable.
    if policy_bucket == "not_released_or_unverifiable" or "not released" in reason:
        return WEIGHTS_MISSING

    # 5) Custom loader required (DEIM / DEIMv2 / RT-DETRv4 etc.).
    if "custom loader" in reason:
        return CUSTOM_LOADER_REQUIRED

    # 6) Partial implementation.
    if implementation_status == "partial":
        return PARTIAL_IMPLEMENTATION_BLOCKED

    # 7) Catalog-only: stub engine, unregistered engine, or external-API-only.
    if (
        implementation_status == "stub"
        or not engine_registered
        or engine == "_stub"
        or policy_bucket == "external_api_only_terms_required"
    ):
        return CATALOG_ONLY_ENGINE_NOT_WIRED

    # 8) Train-ready (wired + reloaded-checkpoint predict).
    if train_ready and inference_ready:
        return (
            TRAIN_READY_LIVE if live_train_verified else TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION
        )

    # 9) Inference-ready. A model that was live-tested and PASSED earns its
    #    task-specific *_LIVE state; one that was live-tested and FAILED earns the
    #    honest blocker recorded by the live matrix (never an optimistic
    #    "derived"); one that was not live-tested stays explicitly derived.
    if inference_ready:
        if live_inference_verified:
            return _LIVE_STATE_FOR_TASK.get(task, INFERENCE_READY_LIVE)
        if live_inference_blocker:
            return live_inference_blocker
        return INFERENCE_READY_DERIVED_NEEDS_LIVE_CONFIRMATION

    # 10) Fallback — should be rare; flags a model the rules did not classify.
    return UNKNOWN_REVIEW_REQUIRED


__all__ = [
    "CATALOG_ONLY_ENGINE_NOT_WIRED",
    "CORRESPONDENCE_READY_LIVE",
    "CUSTOM_LOADER_REQUIRED",
    "DEPENDENCY_MISSING",
    "EMBEDDING_READY_LIVE",
    "GATED_TOKEN_REQUIRED",
    "INFERENCE_READY_DERIVED_NEEDS_LIVE_CONFIRMATION",
    "INFERENCE_READY_LIVE",
    "LICENSE_BLOCKED",
    "LIVE_READY_STATES",
    "NON_COMMERCIAL_BLOCKED",
    "OOM_BLOCKED",
    "OPEN_VOCAB_READY_LIVE",
    "PARTIAL_IMPLEMENTATION_BLOCKED",
    "READINESS_STATES",
    "SEGMENTATION_READY_LIVE",
    "TASK_NOT_SUPPORTED",
    "TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION",
    "TRAIN_READY_LIVE",
    "UNKNOWN_REVIEW_REQUIRED",
    "UPSTREAM_CRASH",
    "VLM_READY_LIVE",
    "WEIGHTS_MISSING",
    "anastig_visibility",
    "classify_license",
    "coarse_readiness",
    "compute_readiness_state",
    "is_copyleft",
    "is_noncommercial",
]
