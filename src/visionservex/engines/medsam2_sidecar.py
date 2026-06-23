# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""MedSAM2 — dependency-gated research-only expert sidecar (honest stub).

MedSAM2 (https://github.com/bowang-lab/MedSAM2) is a video/volumetric promptable
medical segmentation model built on Meta's SAM 2. VisionServeX does **not** ship
a runnable MedSAM2 path, and this engine never fabricates one:

* The published MedSAM2 checkpoints are **non-commercial** (medical dataset
  provenance) — see ``visionservex.licensing.policy`` (``noncommercial_restricted``).
  MedSAM2 must never be labelled commercial-safe or offered as a default.
* The upstream checkpoints are raw SAM 2 ``.pt`` files (not HF ``transformers``
  format), so the ``sam_hf`` / ``sam2_hf`` engines cannot load them; a native
  ``build_sam2`` predictor from the upstream repo is required.

This engine is therefore a *truthful, dependency-gated skeleton*: when the
optional sidecar dependency (``sam2``) is absent it raises a structured
:class:`MissingDependencyError` with an actionable install hint; when the
dependency is present it still raises a structured error because the native
predictor adapter + (non-commercial) checkpoint are intentionally not wired in
core. It NEVER returns mock output as if it were real (mock fallback only happens
when the operator explicitly enables ``models.allow_mock_fallback``, and even
then the result is loudly marked as mock — inherited from :class:`StubEngine`).

When/if a real adapter is added, replace :meth:`_real_load`/``predict`` here and
flip the registry entry to ``implementation_status: wired`` — not before.
"""

from __future__ import annotations

from visionservex.engines._stub import StubEngine
from visionservex.engines.base import MissingDependencyError
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry

#: Required upstream module. MedSAM2 is built on Meta SAM 2 (`sam2`).
MEDSAM2_REQUIRED_MODULES: tuple[str, ...] = ("sam2",)

#: Optional-extra marker used in install hints (see pyproject ``[medsam2]``).
MEDSAM2_INSTALL_EXTRA = "medsam2"

#: Single source of truth for the non-commercial caveat surfaced everywhere.
MEDSAM2_NONCOMMERCIAL_NOTE = (
    "MedSAM2 weights are non-commercial (medical dataset provenance); research "
    "and education only — NOT for diagnosis and NOT commercial-safe."
)

#: Honest install/setup hint shown in structured errors.
MEDSAM2_INSTALL_HINT = (
    "MedSAM2 is a research-only expert sidecar. Install the upstream stack in an "
    "isolated environment: `git clone https://github.com/bowang-lab/MedSAM2 && "
    "cd MedSAM2 && pip install -e .` (provides the `sam2` package), then obtain a "
    "MedSAM2 checkpoint from the upstream repo. " + MEDSAM2_NONCOMMERCIAL_NOTE
)


class MedSAM2SidecarEngine(StubEngine):
    """Dependency-gated MedSAM2 sidecar. Honest failure, never fake inference."""

    real_install_extra = MEDSAM2_INSTALL_EXTRA
    real_modules = MEDSAM2_REQUIRED_MODULES
    backend_label = "medsam2_sidecar"

    def _install_hint(self) -> str:
        # Override the generic StubEngine hint with a precise, honest one.
        return MEDSAM2_INSTALL_HINT

    def _real_load(self, *, device: str, precision: str) -> None:
        # Reached only when `sam2` imports successfully. We deliberately do not
        # ship the native predictor adapter or the (non-commercial) checkpoint in
        # core, so fail with a precise, structured error rather than pretending.
        raise MissingDependencyError(
            "MedSAM2 sidecar dependency is present but the native MedSAM2 predictor "
            "adapter and checkpoint are not wired in this VisionServeX build. "
            + MEDSAM2_NONCOMMERCIAL_NOTE,
            install_hint=MEDSAM2_INSTALL_HINT,
        )


def probe_medsam2_availability() -> dict[str, object]:
    """Return a structured, side-effect-free availability report for MedSAM2.

    Used by the medical CLI and tests to describe *why* MedSAM2 is not runnable
    without importing heavy dependencies or downloading anything.
    """
    import importlib.util

    def _present(name: str) -> bool:
        # find_spec returns None for an absent top-level module, but can raise on a
        # broken/partial install — treat any failure as "absent".
        try:
            return importlib.util.find_spec(name) is not None
        except Exception:
            return False

    missing = [m for m in MEDSAM2_REQUIRED_MODULES if not _present(m)]
    return {
        "model_id": "medsam2",
        "runnable": False,  # never runnable in core — research-only sidecar
        "runtime_status": "expert_sidecar",
        "commercial_safe": False,
        "required_modules": list(MEDSAM2_REQUIRED_MODULES),
        "missing_modules": missing,
        "structured_error_code": "MEDSAM2_REQUIRED" if missing else "MEDSAM2_CHECKPOINT_UNVERIFIED",
        "install_hint": MEDSAM2_INSTALL_HINT,
        "license_note": MEDSAM2_NONCOMMERCIAL_NOTE,
    }


def _factory(entry: ModelEntry) -> MedSAM2SidecarEngine:
    return MedSAM2SidecarEngine(entry)


register_engine("medsam2_sidecar", _factory)

__all__ = [
    "MEDSAM2_INSTALL_HINT",
    "MEDSAM2_NONCOMMERCIAL_NOTE",
    "MEDSAM2_REQUIRED_MODULES",
    "MedSAM2SidecarEngine",
    "probe_medsam2_availability",
]
